"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import csv
import json
import os
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.workload_generator import trace

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
REQUEST_RECORD: List[Tuple[int, float, float, int, int]] = []
REQUEST_ARRIVAL_TS: List[float] = []
REQUEST_FINISH_TS: List[float] = []
# Every completed request; unavailable server metrics remain None (empty CSV cells).
REQUEST_BREAKDOWN = []
BREAKDOWN_STAGES = ("adapter_grouping", "request_collection", "model_compute",
                    "output_reorder", "adapter_adjustment", "worker_other")
ADAPTER_RANK_FIELDS = ("lora_rank", "qkv_rank", "o_rank", "gate_up_rank", "down_rank")
start_ts = [0]

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    force_output_len: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    if "ShareGPT" in dataset_path:
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data for data in dataset
            if len(data["conversations"]) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
    else:
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data for data in dataset
            if len(data["instruction"]) + len(data["input"]) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["instruction"] + data["input"], data["output"])
            for data in dataset
        ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > args.max_len:
            # Prune too long sequences.
            continue
        if args.ratio != 1:
            tot_len = min((prompt_len + output_len) * args.ratio, args.max_len)
            prompt_len = int(prompt_len / (prompt_len + output_len) * tot_len)
            output_len = tot_len - prompt_len
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    if force_output_len > 0:
        ret_requests = []
        output_lens = [t[2] for t in sampled_requests]
        max_output_len = max(output_lens)
        divide = max_output_len // force_output_len
        print(max_output_len, divide)

        for prompt, prompt_len, output_len in sampled_requests:
            output_len = output_len // divide + 1
            ret_requests.append((prompt, prompt_len, output_len))
        
        sampled_requests = ret_requests

    prompt_lens = [t[1] for t in sampled_requests]
    output_lens = [t[2] for t in sampled_requests]
    max_output_len = max(output_lens)
    print(f"avg_prompt_len: {sum(prompt_lens) / len(prompt_lens)},",
          f"avg_output_len: {sum(output_lens) / len(output_lens)},",
          f"max_output_len: {max_output_len}")
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    num_models: int,
    workload: List[Tuple[int, float]],
) -> AsyncGenerator[Tuple[int, int, Tuple[str, int, int]], None]:
    input_requests = iter(input_requests)
    model_cnt = [0] * num_models
    for model_id, interval in workload:
        model_cnt[model_id] += 1
    print("request model cnt:", model_cnt)
    idx = 0
    for request_id, request in enumerate(input_requests):
        model_id, interval = workload[idx]
        idx += 1
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        if request_id < len(REQUEST_ARRIVAL_TS):
            REQUEST_ARRIVAL_TS[request_id] = time.time()
        yield request_id, model_id, request

        # if request_rate == float("inf"):
        #     # If the request rate is infinity, then we don't need to wait.
        #     continue
        # Sample the request interval from the exponential distribution.
        # interval = np.random.exponential(1.0 / request_rate)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    model_id: int,
    request_id: int,
    is_first: bool = False,
) -> None:
    request_start_time = time.time()
    request_start_clock = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        if is_first:
            start_ts[0] = time.time()
        pload = {
            "prompt": prompt,
            "model_id": model_id,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
            "ts": start_ts[0] if is_first else 0,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=12 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            ttft = None
            async with session.post(api_url, headers=headers, json=pload) as response:
                response.raise_for_status()
                output, ttft = await read_response(response, backend == "vllm")

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.time()
    if request_id < len(REQUEST_FINISH_TS):
        REQUEST_FINISH_TS[request_id] = request_end_time
    request_latency = time.perf_counter() - request_start_clock
    tokens_source = "server" if "output_tokens" in output else "requested"
    output_len = output.get("output_tokens", output_len)
    prompt_len = output.get("prompt_tokens", prompt_len)
    if output_len <= 0:
        raise ValueError("Server returned no output tokens; cannot normalize latency")
    breakdown = output.get("latency_breakdown")
    record = {"request_id": request_id, "model_id": model_id,
              "prompt_tokens": prompt_len, "output_tokens": output_len,
              "total_tokens": prompt_len + output_len, "tokens_source": tokens_source,
              "latency_sec": request_latency,
              "processing_time_sec": output.get("processing_time_sec"),
              "ttft_sec": ttft, "profile_available": bool(breakdown)}
    ranks = output.get("adapter_ranks") or {}
    record.update({name: ranks.get(name) for name in ADAPTER_RANK_FIELDS})
    record.update({name + "_sec": None for name in BREAKDOWN_STAGES + ("unattributed",)})
    if breakdown:
        record.update({name + "_sec": breakdown.get(name, 0.0)
                       for name in BREAKDOWN_STAGES})
        # Keep this signed: a negative value exposes inconsistent measurements.
        record["unattributed_sec"] = request_latency - sum(
            record[name + "_sec"] for name in BREAKDOWN_STAGES)
    REQUEST_BREAKDOWN.append(record)
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    REQUEST_RECORD.append((model_id, request_start_time, request_end_time, prompt_len, output_len))


async def read_response(response, streaming):
    """Decode NUL-delimited vLLM frames independently of HTTP chunk boundaries.

    TTFT is the server's active execution duration to its first generated token.
    Never substitute client arrival latency, which includes queueing and network.
    """
    pending = bytearray()
    output = None
    ttft = None
    async for chunk, _ in response.content.iter_chunks():
        pending.extend(chunk)
        while streaming and b"\0" in pending:
            frame, _, remainder = pending.partition(b"\0")
            pending = bytearray(remainder)
            if not frame:
                continue
            output = json.loads(frame)
            if ttft is None:
                ttft = output.get("ttft_sec")
    if pending.strip():
        output = json.loads(pending)
        ttft = output.get("ttft_sec")
    if output is None:
        raise ValueError("Server returned an empty generation response")
    return output, ttft


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    workload: List[Tuple[int, float]],
) -> None:
    tasks: List[asyncio.Task] = []
    is_first = True
    async for request in get_request(input_requests, args.num_models, workload):
        request_id, model_id, (prompt, prompt_len, output_len) = request
        task = asyncio.create_task(send_request(backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search, model_id,
                                                request_id, is_first=is_first))
        is_first = False
        tasks.append(task)
        # model_id = (model_id + 1) % args.num_models
    await asyncio.gather(*tasks)


def collect_midpoint_latency(record_ts: float) -> List[Tuple[int, str, float]]:
    records: List[Tuple[int, str, float]] = []
    for request_id, arrival_ts in enumerate(REQUEST_ARRIVAL_TS):
        finish_ts = REQUEST_FINISH_TS[request_id]
        if arrival_ts < 0 or arrival_ts > record_ts:
            status = "not_arrived"
            latency = -1.0
        elif finish_ts > 0 and finish_ts <= record_ts:
            status = "finished"
            latency = record_ts - arrival_ts
        else:
            status = "waiting_or_running"
            latency = record_ts - arrival_ts
        records.append((request_id, status, latency))
    return records


def write_midpoint_latency(
    output_path: str,
    record_ts: float,
    records: List[Tuple[int, str, float]],
) -> None:
    with open(output_path, "w") as file:
        file.write("request_id,status,latency_sec,arrival_ts,finish_ts,record_ts\n")
        for request_id, status, latency in records:
            arrival_ts = REQUEST_ARRIVAL_TS[request_id]
            finish_ts = REQUEST_FINISH_TS[request_id]
            latency_str = "" if latency < 0 else f"{latency:.6f}"
            file.write(f"{request_id},{status},{latency_str},{arrival_ts:.6f},{finish_ts:.6f},{record_ts:.6f}\n")


def report_latency_breakdown(output_path=None):
    profiled = [row for row in REQUEST_BREAKDOWN if row.get("profile_available", True)]
    if not profiled:
        print("Latency breakdown unavailable: restart the server with FUSELORA_PROFILE=1.")
    else:
        print(f"Synchronized worker breakdown: {len(profiled)}/{len(REQUEST_LATENCY)} requests; "
          "full batch exposure per request, selected TP rank, PP=1.")
        print("Profiling synchronizes CUDA stages and changes stream overlap/performance.")
    for name in ("processing_time_sec", "ttft_sec"):
        values = [row[name] for row in REQUEST_BREAKDOWN if row.get(name) is not None]
        if values:
            print(f"  {name}: mean={np.mean(values):.6f} s/request, "
                  f"p90={np.percentile(values, 90):.6f} s ({len(values)} requests)")
    stages = BREAKDOWN_STAGES + ("unattributed",)
    for name in stages:
        if not profiled:
            break
        values = [row[name + "_sec"] for row in profiled]
        per_token = [row[name + "_sec"] * 1000 / row["output_tokens"]
                     for row in profiled]
        print(f"  {name}: mean={np.mean(values):.6f} s/request, "
              f"mean={np.mean(per_token):.6f} ms/output-token")
    if any(row["unattributed_sec"] < -1e-6 for row in profiled):
        print("Warning: negative unattributed time; check timing coverage before interpreting results.")
    if output_path and REQUEST_BREAKDOWN:
        fields = ["request_id", "model_id", "prompt_tokens", "output_tokens", "total_tokens",
                  "tokens_source", "latency_sec", "processing_time_sec", "ttft_sec", "profile_available"]
        fields += list(ADAPTER_RANK_FIELDS)
        fields += [name + "_sec" for name in stages]
        fields += [name + "_ms_per_output_token" for name in stages]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for record in sorted(REQUEST_BREAKDOWN, key=lambda row: row["request_id"]):
                row = dict(record)
                row.update({name + "_ms_per_output_token": (
                    record[name + "_sec"] * 1000 / record["output_tokens"]
                    if record[name + "_sec"] is not None else None)
                            for name in stages})
                writer.writerow(row)
        print(f"Latency breakdown written to: {output_path}")


def write_adapter_ranks(output_path, num_models):
    """One row per observed adapter/rank configuration, including missing IDs.

    Do not infer ranks from client defaults. If an adapter's configuration varies,
    preserve each observed configuration instead of silently overwriting it.
    """
    configurations = {}
    for row in REQUEST_BREAKDOWN:
        ranks = tuple(row.get(name) for name in ADAPTER_RANK_FIELDS)
        key = (row["model_id"], ranks)
        configurations[key] = configurations.get(key, 0) + 1
    observed = {model_id for model_id, _ in configurations}
    for model_id in range(num_models):
        if model_id not in observed:
            configurations[(model_id, (None,) * len(ADAPTER_RANK_FIELDS))] = 0
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model_id", *ADAPTER_RANK_FIELDS,
                                                 "rank_source", "request_count"])
        writer.writeheader()
        for (model_id, ranks), count in sorted(configurations.items(), key=lambda item: item[0][0]):
            row = dict(zip(ADAPTER_RANK_FIELDS, ranks))
            row.update(model_id=model_id, request_count=count,
                       rank_source="server" if all(rank is not None for rank in ranks) else "unavailable")
            writer.writerow(row)
    print(f"Adapter ranks written to: {output_path}")


def main(args: argparse.Namespace):
    print(args)
    REQUEST_LATENCY.clear()
    REQUEST_RECORD.clear()
    REQUEST_BREAKDOWN.clear()
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len)
    REQUEST_ARRIVAL_TS.clear()
    REQUEST_ARRIVAL_TS.extend([-1.0] * len(input_requests))
    REQUEST_FINISH_TS.clear()
    REQUEST_FINISH_TS.extend([-1.0] * len(input_requests))
    maf_trace = trace.Trace(args.trace_name, args.trace_path, args.start_time, args.end_time, need_sort=args.need_sort)
    workload = maf_trace.replay_to_workload(args.num_models, args.num_prompts, arrival_distribution=args.arrival_distribution, interval_minutes=args.interval, tot_rate=args.request_rate, cv=args.cv, map_stride=args.map_stride)

    benchmark_start_time = time.time()
    asyncio.run(benchmark(args.backend, api_url, input_requests, args.best_of,
                          args.use_beam_search, workload))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    output_len_sum = sum([output_len for _, output_len, _ in REQUEST_LATENCY])
    latency_per_token = benchmark_time * 1000 / output_len_sum
    print(f"Latency per token: {latency_per_token:.2f} ms "
          "(benchmark wall time / total output tokens; inverse throughput)")
    report_latency_breakdown(args.breakdown_output)
    ranks_path = args.adapter_ranks_output or (
        os.path.splitext(args.breakdown_output)[0] + "_adapter_ranks.csv")
    write_adapter_ranks(ranks_path, args.num_models)
    
    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    lat_per_output_token = sorted([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    avg_per_output_token_latency = np.mean(lat_per_output_token)
    p90_idx = int(args.num_prompts * 0.9)
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s\n"
          f"p90: {lat_per_output_token[p90_idx]:.2f} s\n")

    p50_mid_latency = float("nan")
    p90_mid_latency = float("nan")
    p95_mid_latency = float("nan")
    p99_mid_latency = float("nan")
    if args.record_midpoint_latency:
        record_ts = benchmark_start_time + benchmark_time * args.midpoint_ratio
        midpoint_records = collect_midpoint_latency(record_ts)
        arrived_latency = [latency for _, status, latency in midpoint_records
                           if status != "not_arrived"]
        if len(arrived_latency) > 0:
            p50_mid_latency = np.percentile(arrived_latency, 50)
            p90_mid_latency = np.percentile(arrived_latency, 90)
            p95_mid_latency = np.percentile(arrived_latency, 95)
            p99_mid_latency = np.percentile(arrived_latency, 99)
            print("Midpoint latency summary: "
                  f"record_ts={record_ts:.6f}, "
                  f"arrived={len(arrived_latency)}/{len(midpoint_records)}, "
                  f"avg={np.mean(arrived_latency):.4f}s, "
                  f"p50={p50_mid_latency:.4f}s, p90={p90_mid_latency:.4f}s, p95={p95_mid_latency:.4f}s, p99={p99_mid_latency:.4f}s")

            samples = [f"#{request_id}:{latency:.3f}"
                       for request_id, status, latency in midpoint_records
                       if status != "not_arrived"]
            if len(samples) > 0:
                print("Midpoint #req:latency (first 20):", ", ".join(samples[:20]))
        else:
            print("Midpoint latency summary: no arrived requests at record time.")

        if args.midpoint_output is not None:
            write_midpoint_latency(args.midpoint_output, record_ts, midpoint_records)
            print(f"Midpoint latency details written to: {args.midpoint_output}")

    if args.output is not None:
        need_header = (not os.path.exists(args.output)) or os.path.getsize(args.output) == 0
        with open(args.output, "a") as file:
            if need_header:
                if args.output_style == 0:
                    file.write("policy,map_stride,avg_per_output_token_latency,p90_per_output_token_latency\n")
                elif args.output_style == 1:
                    file.write("migration_type,ratio,avg_per_output_token_latency\n")
                elif args.output_style == 2:
                    file.write("exec_type,num_models,avg_throughput\n")
                elif args.output_style == 3:
                    file.write("migration_type,request_rate,queueing_delay_sec,adjust_time_sec,solver_time_sec,total_benchmark_time_sec\n")
                elif args.output_style == 4:
                    file.write("exec_type,request_rate,avg_per_output_token_latency,latency_per_token,p50_mid_latency,p90_mid_latency,p95_mid_latency,p99_mid_latency\n")
                elif args.output_style == 5:
                    file.write("replica_id,queueing_ratio\n")
            if args.output_style == 0:
                file.write(f"{args.policy},{args.map_stride},{avg_per_output_token_latency},{lat_per_output_token[p90_idx]}\n")
            elif args.output_style == 1:
                file.write(f"{args.migration_type},{args.ratio},{avg_per_output_token_latency}\n")
            elif args.output_style == 2:
                file.write(f"{args.exec_type},{args.num_models},{args.num_prompts / benchmark_time}\n")
            elif args.output_style == 3:
                output = os.popen(f"bash ./ae_scripts/parse_log.sh ./ae_scripts/logs/{start_ts[0]}.log").read()
                output = output.split('\n')
                queueing_delay = float(output[0].split(':')[1])
                adjust_time = 0.0
                if len(output[1]) > 0:
                    adjust_time = float(output[1].split(':')[1])
                solver_time = 0.0
                if len(output[2]) > 0:
                    solver_time = float(output[2].split(':')[1])
                file.write(f"{args.migration_type},{args.request_rate},{queueing_delay},{adjust_time},{solver_time},{benchmark_time}\n")
            elif args.output_style == 4:
                # 加入标题
                file.write(f"{args.exec_type},{args.request_rate},{avg_per_output_token_latency},{latency_per_token},{p50_mid_latency},{p90_mid_latency},{p95_mid_latency},{p99_mid_latency}\n")
            elif args.output_style == 5:
                output = os.popen(f"bash ./ae_scripts/parse_log.sh ./ae_scripts/logs/{start_ts[0]}.log fig2b").read()
                output = output.split('\n')
                for line in output:
                    fields = line.split(' ')
                    if len(fields) != 3:
                        continue
                    replica_id = int(fields[0])
                    queueing_delay = float(fields[1])
                    exec_time = float(fields[2])
                    queueing_ratio = queueing_delay / (queueing_delay + exec_time)
                    file.write(f"{replica_id},{queueing_ratio}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--output-len", type=int, default=-1)
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--cv", type=float, default=1,
                        help="Coefficient of variation in gamma process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--trace_name', type=str, default='azure_v1', help='')
    parser.add_argument('--trace_path', type=str, default='~/maf/', help='')
    parser.add_argument('--start_time', type=str, default='0.0.0', help='')
    parser.add_argument('--end_time', type=str, default='0.0.60', help='')
    parser.add_argument("--interval", type=int, default=60, help='')
    parser.add_argument('--need_sort', action='store_true', help='')
    parser.add_argument('--map_stride', type=int, default=1, help='')
    parser.add_argument('--ratio', type=float, default=1.0, help='')
    parser.add_argument('--max_len', type=int, default=2048, help='')
    parser.add_argument('--policy', type=str, default='credit', help='')
    parser.add_argument('--migration_type', type=int, default=3, help='')
    parser.add_argument('--exec_type', type=int, default=3, help='')
    parser.add_argument('--log_path', type=str, default=None, help='')
    parser.add_argument('--output', type=str, default=None, help='')
    parser.add_argument('--output_style', type=int, default=0, help='')
    parser.add_argument('--arrival_distribution', type=str, default='gamma', help='')
    parser.add_argument('--record_midpoint_latency', action='store_true', default=True,
                        help='Record per-request latency at benchmark midpoint.')
    parser.add_argument('--midpoint_ratio', type=float, default=0.5,
                        help='Record timestamp = start + benchmark_time * midpoint_ratio.')
    parser.add_argument('--breakdown-output', '--request-output', type=str, default='request_metrics.csv',
                        help='Per-request metrics CSV (default: request_metrics.csv); '
                             'worker stages require server FUSELORA_PROFILE=1.')
    parser.add_argument('--midpoint_output', type=str, default=None,
                        help='Optional CSV path for midpoint per-request latencies.')
    parser.add_argument('--adapter-ranks-output', type=str, default=None,
                        help='Adapter rank CSV; defaults to <request CSV stem>_adapter_ranks.csv.')
    args = parser.parse_args()
    main(args)
