# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


def _build_balanced_adapter_partitions(
    adapter_mapping: torch.Tensor,
    num_adapter: int,
    num_parts: int,
) -> List[List[int]]:
    # requests per adapter in current batch
    req_cnt = adapter_mapping.sum(dim=0).to(dtype=torch.float32)
    req_cnt_cpu = req_cnt.detach().to("cpu")

    sorted_adapter_ids = torch.argsort(req_cnt_cpu, descending=True).tolist()
    part_num = max(1, min(int(num_parts), num_adapter))
    partitions: List[List[int]] = [[] for _ in range(part_num)]
    part_loads = [0.0 for _ in range(part_num)]

    # greedy load balancing by current request counts
    for adapter_id in sorted_adapter_ids:
        target_part = min(range(part_num), key=lambda i: part_loads[i])
        partitions[target_part].append(adapter_id)
        part_loads[target_part] += float(req_cnt_cpu[adapter_id].item())

    return [p for p in partitions if p]


def _partitioned_all_lora(
    x: torch.Tensor,
    adapter_mapping: torch.Tensor,
    lora_As: torch.Tensor,
    lora_Bs: torch.Tensor,
    in_features: int,
    out_features: int,
    partition_m: int,
    r_factor: int = 1,
) -> torch.Tensor:
    device = x.device
    adapter_ids = torch.argmax(adapter_mapping, dim=1).to(dtype=torch.long)
    num_adapter = lora_As.size(0)
    r_total = lora_As.size(2)
    r = r_total // r_factor

    partitions = _build_balanced_adapter_partitions(
        adapter_mapping=adapter_mapping,
        num_adapter=num_adapter,
        num_parts=partition_m,
    )

    # fallback to single chunk path
    if len(partitions) <= 1:
        A_all = lora_As.permute(1, 0, 2).reshape(in_features, num_adapter * r_total)
        B_all = lora_Bs.reshape(num_adapter * r_total, r_factor * out_features)
        H = x @ A_all
        rank_offsets = adapter_ids * r_total
        idx = rank_offsets.unsqueeze(1) + torch.arange(r_total, device=device)
        H_selected = torch.gather(H, 1, idx)
        B_all = B_all.view(num_adapter, r_total, r_factor * out_features)
        B_selected = B_all[adapter_ids]
        return torch.bmm(H_selected.unsqueeze(1), B_selected).squeeze(1)

    y = torch.zeros(
        (x.size(0), r_factor * out_features),
        dtype=x.dtype,
        device=device,
    )

    use_streams = x.is_cuda and len(partitions) > 1
    streams = [torch.cuda.Stream(device=device) for _ in partitions] if use_streams else [None] * len(partitions)

    for part_idx, adapter_group in enumerate(partitions):
        group_ids = torch.tensor(adapter_group, device=device, dtype=torch.long)
        local_id_map = torch.full((num_adapter,), -1, device=device, dtype=torch.long)
        local_id_map[group_ids] = torch.arange(group_ids.numel(), device=device, dtype=torch.long)

        local_ids_all = local_id_map[adapter_ids]
        token_mask = local_ids_all >= 0
        if not torch.any(token_mask):
            continue

        def _run_chunk() -> None:
            local_x = x[token_mask]
            local_ids = local_ids_all[token_mask]

            part_As = lora_As[group_ids]
            part_Bs = lora_Bs[group_ids]
            k = part_As.size(0)

            A_all = part_As.permute(1, 0, 2).reshape(in_features, k * r_total)
            B_all = part_Bs.reshape(k * r_total, r_factor * out_features)

            H = local_x @ A_all
            rank_offsets = local_ids * r_total
            idx = rank_offsets.unsqueeze(1) + torch.arange(r_total, device=device)
            H_selected = torch.gather(H, 1, idx)

            B_all = B_all.view(k, r_total, r_factor * out_features)
            B_selected = B_all[local_ids]
            y[token_mask] = torch.bmm(H_selected.unsqueeze(1), B_selected).squeeze(1)

        if streams[part_idx] is None:
            _run_chunk()
        else:
            with torch.cuda.stream(streams[part_idx]):
                _run_chunk()

    if use_streams:
        current_stream = torch.cuda.current_stream(device=device)
        for stream in streams:
            current_stream.wait_stream(stream)

    return y

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class LoraLayer:
    lora_stream = torch.cuda.Stream()
    assert lora_stream != torch.cuda.current_stream()

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.merge_lora = nn.ModuleDict({})
        self.merged_mapping = {}
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            # self.merge_lora.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.out_features, bias=False)}))
            # self.merged_mapping[adapter_name] = False
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features), dtype=self.weight.dtype, device=self.weight.device)
            weight_B = torch.randn((self.out_features, r), dtype=self.weight.dtype, device=self.weight.device)
            self.lora_embedding_A.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_A)}))
            self.lora_embedding_B.update(nn.ParameterDict({adapter_name: nn.Parameter(weight_B)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class Linear(nn.Linear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        params_dtype = None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # self.partition_m = max(1, int(kwargs.pop("m", kwargs.pop("lora_partition_m", 1))))
        self.partition_m = 3
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        nn.Linear.__init__(self, in_features, out_features, **kwargs, dtype=params_dtype)
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_As = None
        self.lora_Bs = None
        self.in_features = in_features
        self.out_features = out_features
        self.max_r = r
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.kwargs = kwargs

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        # Actual trainable parameters
        if r > self.max_r:
            padding_As = torch.zeros(self.lora_As.shape[0], self.in_features, r - self.max_r).cuda()
            padding_Bs = torch.zeros(self.lora_Bs.shape[0], r - self.max_r, self.out_features).cuda()
            self.lora_As = torch.cat((self.lora_As, padding_As), dim=-1)
            self.lora_Bs = torch.cat((self.lora_Bs, padding_Bs), dim=1)
            self.max_r = r
            lora_A = torch.randn(1, self.in_features, r).cuda()
            lora_B = torch.randn(1, r, self.out_features).cuda()
        else:
            lora_A = torch.randn(1, self.in_features, self.max_r).cuda()
            lora_B = torch.randn(1, self.max_r, self.out_features).cuda()
            lora_A[:, :, r:] = 0
            lora_B[:, r:, :] = 0
        if r > 0:
            self.scaling[adapter_name] = lora_alpha / r
        if self.lora_As == None:
            self.lora_As = lora_A
            self.lora_Bs = lora_B
        else:
            self.lora_As = torch.cat((self.lora_As, lora_A), dim=0)
            self.lora_Bs = torch.cat((self.lora_Bs, lora_B), dim=0)
            
        self.to(self.weight.device)

    # def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor):
    #     return torch.einsum('bk, bi, ikr, ird->bd',
    #             x,
    #             adapter_mapping,
    #             self.lora_As,
    #             self.lora_Bs
    #         )
    def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor):
        return _partitioned_all_lora(
            x=x,
            adapter_mapping=adapter_mapping,
            lora_As=self.lora_As,
            lora_Bs=self.lora_Bs,
            in_features=self.in_features,
            out_features=self.out_features,
            partition_m=self.partition_m,
            r_factor=1,
        )

    def forward(self, x: torch.Tensor, adapter_mapping: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias) + self.all_lora(x, adapter_mapping)
        return result
    
class QKV_Linear(nn.Linear):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        qkv_r: List[int] = {0, 0, 0},
        qkv_lora_alpha: List[int] = {1, 1, 1},
        qkv_lora_dropout: List[float] = {0.0, 0.0, 0.0},
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        params_dtype = None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # self.partition_m = max(1, int(kwargs.pop("m", kwargs.pop("lora_partition_m", 1))))
        self.partition_m = 2
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        nn.Linear.__init__(self, in_features, 3 * out_features, **kwargs, dtype=params_dtype)
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_As = None
        self.lora_Bs = None
        self.in_features = in_features
        self.out_features = out_features
        self.max_r = max(qkv_r)
        self.weight.requires_grad = False
        self.kwargs = kwargs

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, qkv_r, qkv_lora_alpha, qkv_lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def update_layer(self, adapter_name, qkv_r, qkv_lora_alpha, qkv_lora_dropout, init_lora_weights):
        assert len(qkv_r) == len(qkv_lora_alpha) == len(qkv_lora_dropout) == 3
        self.r[adapter_name] = qkv_r
        self.lora_alpha[adapter_name] = qkv_lora_alpha
        self.scaling[adapter_name] = [qkv_lora_alpha[i] / qkv_r[i] if qkv_r[i] != 0 else 0 for i in range(3)]

        r = max(qkv_r)
        if r > self.max_r:
            padding_As = torch.zeros(self.lora_As.shape[0], self.in_features, 3 * (r - self.max_r)).cuda()
            padding_Bs = torch.zeros(self.lora_Bs.shape[0], 3 * (r - self.max_r), 3 * self.out_features).cuda()
            self.lora_As = torch.cat((self.lora_As, padding_As), dim=-1)
            self.lora_Bs = torch.cat((self.lora_Bs, padding_Bs), dim=1)
            self.max_r = r
            lora_A = torch.randn(1, self.in_features, 3 * r).cuda()
            lora_B = torch.randn(1, r, 3 * self.out_features).cuda()
            lora_B = torch.cat((lora_B, lora_B, lora_B), dim=1)
        else:
            lora_A = torch.randn(1, self.in_features, 3 * self.max_r).cuda()
            lora_B = torch.randn(1, self.max_r, 3 * self.out_features).cuda()
            lora_B = torch.cat((lora_B, lora_B, lora_B), dim=1)
        
        for i in range(3):
            lora_A[:, :, qkv_r[i]:(i+1)*r] = 0
            lora_B[:, qkv_r[i]:(i+1)*r, :] = 0
        if self.lora_As == None:
            self.lora_As = lora_A
            self.lora_Bs = lora_B
        else:
            self.lora_As = torch.cat((self.lora_As, lora_A), dim=0)
            self.lora_Bs = torch.cat((self.lora_Bs, lora_B), dim=0)
        self.to(self.weight.device)

    # def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor):
    #     return torch.einsum('bk, bi, ikr, ird->bd',
    #             x,
    #             adapter_mapping,
    #             self.lora_As,
    #             self.lora_Bs
    #         )

    def all_lora(self, x: torch.Tensor, adapter_mapping: torch.Tensor):
        return _partitioned_all_lora(
            x=x,
            adapter_mapping=adapter_mapping,
            lora_As=self.lora_As,
            lora_Bs=self.lora_Bs,
            in_features=self.in_features,
            out_features=self.out_features,
            partition_m=self.partition_m,
            r_factor=3,
        )

    def forward(self, x: torch.Tensor, adapter_mapping):
        result = F.linear(x, self.weight, bias=self.bias) + self.all_lora(x, adapter_mapping)
        return result



class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_embedding_B[self.active_adapter] @ self.lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lora_embedding_B[self.active_adapter].weight
                        @ self.lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x)

        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding)
        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Conv2d.reset_parameters(self)
        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data += (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[self.active_adapter]
                )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            if self.weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                self.weight.data -= (
                    self.lora_B[self.active_adapter].weight.squeeze(3).squeeze(2)
                    @ self.lora_A[self.active_adapter].weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3) * self.scaling[self.active_adapter]
            else:
                # conv2d 3x3
                self.weight.data += (
                    F.conv2d(
                        self.lora_A[self.active_adapter].weight.permute(1, 0, 2, 3),
                        self.lora_B[self.active_adapter].weight,
                    ).permute(1, 0, 2, 3)
                    * self.scaling[self.active_adapter]
                )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        result = result.to(previous_dtype)

        return result