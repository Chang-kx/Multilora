"""Grouped LoRA projections shared by native and PEFT linear layers."""
from typing import List

import torch

from vllm.latency_profile import phase
from vllm.lora_grouping import (balanced_adapter_partitions,
                                current_batch_partitions)


def _build_balanced_adapter_partitions(
    adapter_mapping: torch.Tensor,
    num_adapter: int,
    num_parts: int,
) -> List[List[int]]:
    # Fallback for warmup/standalone calls. Serving supplies request-based
    # partitions once per batch, before prompts are expanded into tokens.
    counts = adapter_mapping.sum(dim=0).detach().to("cpu").tolist()
    return balanced_adapter_partitions(counts[:num_adapter], int(num_parts))


def _partitioned_all_lora(
    x: torch.Tensor,
    adapter_mapping: torch.Tensor,
    lora_As: torch.Tensor,
    lora_Bs: torch.Tensor,
    in_features: int,
    out_features: int,
    partition_m: int = 1,
    r_factor: int = 1,
) -> torch.Tensor:
    device = x.device
    with phase("adapter_grouping"):
        adapter_ids = torch.argmax(adapter_mapping, dim=1).to(dtype=torch.long)
        num_adapter = lora_As.size(0)
        r_total = lora_As.size(2)
        partitions = current_batch_partitions()
        if partitions is None:
            partitions = _build_balanced_adapter_partitions(
                adapter_mapping=adapter_mapping,
                num_adapter=num_adapter,
                num_parts=partition_m,
            )

    if len(partitions) == 1 and len(partitions[0]) == num_adapter:
        with phase("request_collection"):
            A_all = lora_As.permute(1, 0, 2).reshape(in_features, num_adapter * r_total)
            B_all = lora_Bs.reshape(num_adapter, r_total, r_factor * out_features)
            B_selected = B_all[adapter_ids]
        with phase("model_compute"):
            H = x @ A_all
        with phase("request_collection"):
            rank_offsets = adapter_ids * r_total
            idx = rank_offsets.unsqueeze(1) + torch.arange(r_total, device=device)
            H_selected = torch.gather(H, 1, idx)
        with phase("model_compute"):
            return torch.bmm(H_selected.unsqueeze(1), B_selected).squeeze(1)

    with phase("request_collection"):
        y = torch.zeros(
            (x.size(0), r_factor * out_features), dtype=x.dtype, device=device)
        use_streams = x.is_cuda and len(partitions) > 1
        streams = [torch.cuda.Stream(device=device) for _ in partitions] if use_streams else [None] * len(partitions)

    for part_idx, adapter_group in enumerate(partitions):
        with phase("adapter_grouping"):
            group_ids = torch.tensor(adapter_group, device=device, dtype=torch.long)
            local_id_map = torch.full((num_adapter,), -1, device=device, dtype=torch.long)
            local_id_map[group_ids] = torch.arange(group_ids.numel(), device=device, dtype=torch.long)
            local_ids_all = local_id_map[adapter_ids]
            token_mask = local_ids_all >= 0

        def _run_chunk() -> None:
            with phase("request_collection"):
                local_x = x[token_mask]
                local_ids = local_ids_all[token_mask]
                part_As = lora_As[group_ids]
                part_Bs = lora_Bs[group_ids]
                k = part_As.size(0)
                A_all = part_As.permute(1, 0, 2).reshape(in_features, k * r_total)
                B_all = part_Bs.reshape(k, r_total, r_factor * out_features)
                B_selected = B_all[local_ids]
            with phase("model_compute"):
                H = local_x @ A_all
            with phase("request_collection"):
                rank_offsets = local_ids * r_total
                idx = rank_offsets.unsqueeze(1) + torch.arange(r_total, device=device)
                H_selected = torch.gather(H, 1, idx)
            with phase("model_compute"):
                local_y = torch.bmm(H_selected.unsqueeze(1), B_selected).squeeze(1)
            with phase("output_reorder"):
                y[token_mask] = local_y

        if streams[part_idx] is None:
            _run_chunk()
        else:
            # Inputs and masks were produced on the caller's stream.
            streams[part_idx].wait_stream(torch.cuda.current_stream(device=device))
            # These temporary tensors may be released on the caller's stream
            # while the next group is being prepared.
            for tensor in (group_ids, local_ids_all, token_mask):
                tensor.record_stream(streams[part_idx])
            with torch.cuda.stream(streams[part_idx]):
                _run_chunk()

    if use_streams:
        current_stream = torch.cuda.current_stream(device=device)
        for stream in streams:
            current_stream.wait_stream(stream)
    return y

