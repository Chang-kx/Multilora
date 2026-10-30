"""CPU-only checks; load target code without importing the CUDA-only vllm package."""
import ast
import asyncio
import csv
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RANK_FIELDS = ("lora_rank", "qkv_rank", "o_rank", "gate_up_rank", "down_rank")
spec = importlib.util.spec_from_file_location("latency_profile", ROOT / "vllm/latency_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
grouping_spec = importlib.util.spec_from_file_location(
    "latency_test_lora_grouping", ROOT / "vllm/lora_grouping.py")
grouping = importlib.util.module_from_spec(grouping_spec)
sys.modules[grouping_spec.name] = grouping
grouping_spec.loader.exec_module(grouping)


def load_functions(relative_path, names, namespace):
    namespace.setdefault("ADAPTER_RANK_FIELDS", RANK_FIELDS)
    tree = ast.parse((ROOT / relative_path).read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(tree, relative_path, "exec"), namespace)
    return namespace


class LatencyBreakdownTests(unittest.TestCase):
    def test_exclusive_nested_times_and_context_cleanup(self):
        from collections import defaultdict
        totals = defaultdict(float)
        token = profile._ACTIVE.set((lambda: None, totals, []))
        try:
            with patch.object(profile.time, "perf_counter", side_effect=[0, 1, 3, 5]):
                with profile.phase("model_compute"):
                    with profile.phase("request_collection"):
                        pass
            self.assertEqual(dict(totals), {"request_collection": 2, "model_compute": 3})
            with self.assertRaises(RuntimeError):
                with profile.phase("worker_other"):
                    raise RuntimeError("failed")
            self.assertEqual(profile._ACTIVE.get()[2], [])
        finally:
            profile._ACTIVE.reset(token)
        self.assertIsNone(profile._ACTIVE.get())

    def test_worker_attaches_once_without_changing_token_output(self):
        output = {1: types.SimpleNamespace(output_token=42)}
        worker = types.SimpleNamespace(device="cuda:0", parallel_config=types.SimpleNamespace(pipeline_parallel_size=1))

        @profile.profile_worker
        def execute(self):
            with profile.phase("model_compute"):
                pass
            return output

        with patch.dict("os.environ", {"FUSELORA_PROFILE": "0"}):
            self.assertIs(execute(worker), output)
            self.assertFalse(hasattr(output[1], "latency_breakdown"))
        with patch.dict("os.environ", {"FUSELORA_PROFILE": "1"}), patch.object(torch.cuda, "synchronize"):
            self.assertEqual(execute(worker)[1].output_token, 42)
            self.assertEqual(set(output[1].latency_breakdown), {"model_compute", "worker_other"})
            self.assertIsNone(profile._ACTIVE.get())
            worker.parallel_config.pipeline_parallel_size = 2
            with self.assertRaisesRegex(ValueError, "pipeline_parallel_size=1"):
                execute(worker)

    def test_partitioned_lora_matches_per_token_reference(self):
        namespace = load_functions("vllm/model_executor/layers/lora_ops.py",
                                   {"_build_balanced_adapter_partitions", "_partitioned_all_lora"},
                                   {"torch": torch, "List": list, "phase": profile.phase,
                                    "balanced_adapter_partitions": grouping.balanced_adapter_partitions,
                                    "current_batch_partitions": grouping.current_batch_partitions})
        torch.manual_seed(7)
        for adapters, parts, factor in [(1, 4, 1), (4, 1, 1), (4, 4, 1), (4, 3, 3)]:
            with self.subTest(adapters=adapters, parts=parts, factor=factor):
                ids = torch.tensor([0, adapters - 1, 0, 0, adapters - 1])
                mapping = torch.nn.functional.one_hot(ids, adapters).float()
                x = torch.randn(5, 6)
                a = torch.randn(adapters, 6, 2 * factor)
                b = torch.randn(adapters, 2 * factor, 7 * factor)
                expected = torch.stack([x[i] @ a[ids[i]] @ b[ids[i]] for i in range(5)])
                actual = namespace["_partitioned_all_lora"](x, mapping, a, b, 6, 7, parts, factor)
                torch.testing.assert_close(actual, expected)

    def test_scheduler_counts_batch_once_per_group_across_iterations(self):
        tree = ast.parse((ROOT / "vllm/core/scheduler.py").read_text())
        scheduler_class = next(node for node in tree.body
                               if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
        update = next(node for node in scheduler_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "update")
        update.returns = None
        for argument in update.args.args:
            argument.annotation = None
        status = types.SimpleNamespace(RUNNING=1, READY=2)
        namespace = {"time": types.SimpleNamespace(time=lambda: 10.0, perf_counter=lambda: 20.0), "SequenceStatus": status}
        exec(compile(ast.Module(body=[update], type_ignores=[]), "scheduler.update", "exec"), namespace)
        seqs = [types.SimpleNamespace(seq_id=i, status=status.RUNNING,
                                     append_token_id=lambda *args: None) for i in (1, 2)]
        group = types.SimpleNamespace(exec_time=0.0, start_time=9.0,
                                      execution_start_clock=19.5,
                                      get_seqs=lambda status: [seq for seq in seqs if seq.status == status])
        scheduler = types.SimpleNamespace(running=[group], ready=[])
        output = {i: types.SimpleNamespace(parent_seq_id=i, output_token=8, logprobs={},
                                          latency_breakdown={"model_compute": 0.3}) for i in (1, 2)}
        for _ in range(2):
            scheduler.running = [group]
            for seq in seqs:
                seq.status = status.RUNNING
            namespace["update"](scheduler, output)
        self.assertAlmostEqual(group.latency_breakdown["model_compute"], 0.6)
        self.assertAlmostEqual(group.processing_time_sec, 1.0)
        # First-token time is fixed at the first iteration's active time;
        # subsequent decode execution and initial wall-clock waiting are excluded.
        self.assertAlmostEqual(group.ttft_sec, 0.5)

    def test_csv_stages_reconcile_and_missing_profiles_are_not_zero_rows(self):
        stages = ("adapter_grouping", "request_collection", "model_compute", "output_reorder", "adapter_adjustment", "worker_other")
        row = {"request_id": 0, "model_id": 3, "output_tokens": 10, "latency_sec": 2.0}
        row.update({name + "_sec": 0.1 for name in stages})
        row["unattributed_sec"] = 1.4
        namespace = load_functions("benchmarks/benchmark_serving.py", {"report_latency_breakdown"},
                                   {"np": np, "csv": csv, "os": os, "REQUEST_BREAKDOWN": [row],
                                    "REQUEST_LATENCY": [None], "BREAKDOWN_STAGES": stages})
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "profile.csv"
            namespace["report_latency_breakdown"](path)
            with path.open() as file:
                result = next(csv.DictReader(file))
            fields = stages + ("unattributed",)
            self.assertAlmostEqual(sum(float(result[name + "_sec"]) for name in fields), 2.0)
            self.assertAlmostEqual(sum(float(result[name + "_ms_per_output_token"]) for name in fields), 200.0)
            namespace["REQUEST_BREAKDOWN"] = []
            missing = Path(directory) / "missing.csv"
            namespace["report_latency_breakdown"](missing)
            self.assertFalse(missing.exists())

    def test_stream_frames_split_across_chunks_and_empty_text_token(self):
        namespace = load_functions("benchmarks/benchmark_serving.py", {"read_response"},
                                   {"json": json, "time": types.SimpleNamespace(perf_counter=lambda: 12.5)})
        async def chunks():
            frames = [b'{"output_tokens":0}\0{"output_',
                      b'tokens":1,"text":[""],"ttft_sec":0.3}\0{"output_tokens":',
                      b'3,"finished":true,"ttft_sec":0.3}\0']
            for frame in frames:
                yield frame, False
        response = types.SimpleNamespace(content=types.SimpleNamespace(iter_chunks=chunks))
        output, ttft = asyncio.run(namespace["read_response"](response, True))
        self.assertEqual(output["output_tokens"], 3)
        self.assertEqual(ttft, 0.3)

    def test_plain_response_does_not_invent_ttft(self):
        namespace = load_functions("benchmarks/benchmark_serving.py", {"read_response"},
                                   {"json": json})
        async def chunks():
            yield b'{"output_tokens":3}', False
        response = types.SimpleNamespace(content=types.SimpleNamespace(iter_chunks=chunks))
        output, ttft = asyncio.run(namespace["read_response"](response, True))
        self.assertEqual(output["output_tokens"], 3)
        self.assertIsNone(ttft)

    def test_nonstreaming_response_uses_server_execution_ttft(self):
        namespace = load_functions("benchmarks/benchmark_serving.py", {"read_response"},
                                   {"json": json})
        async def chunks():
            yield b'{"output_tokens":3,"ttft_sec":0.15}', False
        response = types.SimpleNamespace(content=types.SimpleNamespace(iter_chunks=chunks))
        _, ttft = asyncio.run(namespace["read_response"](response, False))
        self.assertEqual(ttft, 0.15)

    def test_unprofiled_request_still_written_with_blank_stages(self):
        stages = ("adapter_grouping", "request_collection", "model_compute", "output_reorder", "adapter_adjustment", "worker_other")
        row = {"request_id": 4, "profile_available": False, "output_tokens": 3,
               "prompt_tokens": 5, "total_tokens": 8, "ttft_sec": 0.2,
               "processing_time_sec": 0.5}
        row.update({name + "_sec": None for name in stages + ("unattributed",)})
        namespace = load_functions("benchmarks/benchmark_serving.py", {"report_latency_breakdown"},
                                   {"np": np, "csv": csv, "os": os, "REQUEST_BREAKDOWN": [row],
                                    "REQUEST_LATENCY": [None], "BREAKDOWN_STAGES": stages})
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "nested" / "requests.csv"
            namespace["report_latency_breakdown"](path)
            with path.open() as file:
                saved = next(csv.DictReader(file))
            self.assertEqual(saved["total_tokens"], "8")
            self.assertEqual(saved["ttft_sec"], "0.2")
            self.assertEqual(saved["adapter_grouping_sec"], "")
            self.assertEqual(saved["lora_rank"], "")

    def test_api_serializes_actual_token_ids_and_processing_time(self):
        namespace = load_functions("vllm/entrypoints/api_server.py", {"serialize_output"}, {})
        output = types.SimpleNamespace(prompt="hello", prompt_token_ids=[1, 2, 3],
                                       outputs=[types.SimpleNamespace(text="", token_ids=[4, 5])],
                                       processing_time_sec=0.4, ttft_sec=0.15, finished=True,
                                       latency_breakdown={"model_compute": 0.3})
        result = namespace["serialize_output"](output)
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["total_tokens"], 5)
        self.assertEqual(result["processing_time_sec"], 0.4)
        self.assertEqual(result["ttft_sec"], 0.15)
        self.assertEqual(result["latency_breakdown"]["model_compute"], 0.3)
        self.assertEqual(result["adapter_ranks"], {})
        output.adapter_ranks = dict(zip(RANK_FIELDS, (32, 96, 32, 32, 32)))
        self.assertEqual(namespace["serialize_output"](output)["adapter_ranks"], output.adapter_ranks)

    def test_send_request_records_actual_tokens_and_separate_execution_latency(self):
        stages = ("adapter_grouping", "request_collection", "model_compute", "output_reorder", "adapter_adjustment", "worker_other")
        final = {"output_tokens": 3, "prompt_tokens": 5, "processing_time_sec": 0.7, "ttft_sec": 0.15,
                 "latency_breakdown": {stage: 0.1 for stage in stages},
                 "adapter_ranks": dict(zip(RANK_FIELDS, (32, 96, 32, 32, 32)))}
        async def chunks():
            yield b'{"output_tokens":1}\0', False
            yield json.dumps(final).encode() + b'\0', False

        class Context:
            def __init__(self, value):
                self.value = value

            async def __aenter__(self):
                return self.value

            async def __aexit__(self, *args):
                return False

        response = types.SimpleNamespace(raise_for_status=lambda: None,
                                         content=types.SimpleNamespace(iter_chunks=chunks))
        def post(*args, **kwargs):
            self.assertTrue(kwargs["json"]["stream"])
            return Context(response)
        session = types.SimpleNamespace(post=post)
        clocks = iter([10.0, 11.0])
        namespace = load_functions("benchmarks/benchmark_serving.py", {"send_request", "read_response"}, {
            "time": types.SimpleNamespace(time=lambda: 100.0, perf_counter=lambda: next(clocks)),
            "aiohttp": types.SimpleNamespace(ClientTimeout=lambda **kwargs: None,
                                             ClientSession=lambda **kwargs: Context(session)),
            "json": json, "BREAKDOWN_STAGES": stages, "REQUEST_FINISH_TS": [-1],
            "REQUEST_BREAKDOWN": [], "REQUEST_LATENCY": [], "REQUEST_RECORD": [], "start_ts": [0]})
        asyncio.run(namespace["send_request"]("vllm", "http://test/generate", "hi", 9, 10,
                                              1, False, 2, 0))
        row = namespace["REQUEST_BREAKDOWN"][0]
        self.assertEqual((row["prompt_tokens"], row["output_tokens"], row["total_tokens"]), (5, 3, 8))
        self.assertAlmostEqual(row["ttft_sec"], 0.15)
        self.assertEqual(row["processing_time_sec"], 0.7)
        self.assertEqual(row["latency_sec"], 1.0)
        self.assertAlmostEqual(row["unattributed_sec"], 0.4)
        self.assertEqual(row["tokens_source"], "server")
        self.assertEqual(row["lora_rank"], 32)
        self.assertEqual(row["qkv_rank"], 96)

    def test_adapter_rank_csv_deduplicates_and_preserves_missing_and_changed_ranks(self):
        rank16 = dict(zip(RANK_FIELDS, (16, 48, 16, 16, 16)))
        rank32 = dict(zip(RANK_FIELDS, (32, 96, 32, 32, 32)))
        records = [{"model_id": 0, **rank16}, {"model_id": 0, **rank16},
                   {"model_id": 0, **rank32}, {"model_id": 1}]
        namespace = load_functions("benchmarks/benchmark_serving.py", {"write_adapter_ranks"},
                                   {"csv": csv, "os": os, "REQUEST_BREAKDOWN": records})
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "ranks" / "adapters.csv"
            namespace["write_adapter_ranks"](path, 3)
            with path.open() as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["model_id"] for row in rows], ["0", "0", "1", "2"])
        self.assertEqual(rows[0]["lora_rank"], "16")
        self.assertEqual(rows[0]["request_count"], "2")
        self.assertEqual(rows[1]["qkv_rank"], "96")
        self.assertEqual(rows[1]["rank_source"], "server")
        self.assertEqual(rows[2]["lora_rank"], "")
        self.assertEqual(rows[2]["rank_source"], "unavailable")
        self.assertEqual(rows[2]["request_count"], "1")
        self.assertEqual(rows[3]["request_count"], "0")

    def test_engine_records_configured_rank_on_migratable_request_group(self):
        tree = ast.parse((ROOT / "vllm/engine/llm_engine.py").read_text())
        engine_class = next(node for node in tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "LLMEngine")
        method = next(node for node in engine_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "add_request")
        namespace = {
            "Sequence": lambda *args: object(),
            "SequenceGroup": lambda *args, **kwargs: types.SimpleNamespace(model_id=kwargs["model_id"]),
            "ExecType": types.SimpleNamespace(LORA=3),
        }
        module = ast.Module(body=ast.parse("from __future__ import annotations").body + [method],
                            type_ignores=[])
        exec(compile(module, "engine.add_request", "exec"), namespace)
        groups = []
        engine = types.SimpleNamespace(exec_type=3, lora_config=types.SimpleNamespace(max_r=32),
                                       cache_config=types.SimpleNamespace(block_size=16), seq_counter=iter(range(10)),
                                       scheduler=types.SimpleNamespace(add_seq_group=groups.append))
        namespace["add_request"](engine, "req", "hi", types.SimpleNamespace(best_of=1),
                                  prompt_token_ids=[1, 2], arrival_time=0.0, model_id=7)
        migrated = copy.deepcopy(groups[0])
        self.assertEqual(migrated.model_id, 7)
        self.assertEqual(migrated.adapter_ranks, dict(zip(RANK_FIELDS, (32, 96, 32, 32, 32))))
        engine.exec_type = 1
        namespace["add_request"](engine, "req2", "hi", types.SimpleNamespace(best_of=1),
                                  prompt_token_ids=[1, 2], arrival_time=0.0)
        self.assertFalse(hasattr(groups[1], "adapter_ranks"))


if __name__ == "__main__":
    unittest.main()
