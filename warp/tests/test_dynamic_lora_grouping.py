"""CPU policy/integration checks; run with unittest to avoid CUDA conftest."""
import argparse
import ast
from contextlib import nullcontext
from functools import wraps
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dynamic_grouping_test", ROOT / "vllm/lora_grouping.py")
grouping = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = grouping
spec.loader.exec_module(grouping)


def load_nodes(relative, namespace, class_name=None, names=None):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes
                     if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in nodes if getattr(node, "name", None) in names]
    tree.body = ast.parse("from __future__ import annotations").body + nodes
    exec(compile(tree, relative, "exec"), namespace)
    return namespace


def ops_namespace():
    return load_nodes("vllm/model_executor/layers/lora_ops.py", {
        "torch": torch, "phase": lambda name: nullcontext(),
        "balanced_adapter_partitions": grouping.balanced_adapter_partitions,
        "current_batch_partitions": grouping.current_batch_partitions,
    }, names={"_build_balanced_adapter_partitions", "_partitioned_all_lora"})


class GroupingPolicyTests(unittest.TestCase):
    def complete(self, policy, counts, elapsed=1.0):
        plan = policy.select(counts)
        policy.finish(plan, elapsed)
        return plan

    def test_defaults_and_first_search_find_first_feasible_group(self):
        policy = grouping.DynamicLoRaGrouping()
        self.assertEqual((policy.c_full, policy.threshold), (2000, 1600))
        plan = policy.select([5] * 20)
        self.assertEqual((plan.num_groups, plan.compute), (2, 1000))
        self.assertEqual(policy.completed_batches, 0)
        self.assertEqual(policy.select([1] * 30).compute, 900)
        example = grouping.DynamicLoRaGrouping(20, 20, 0.8).select([1] * 30)
        self.assertEqual((example.num_groups, example.compute), (3, 300))
        self.assertEqual([len(part) for part in example.partitions], [10, 10, 10])

    def test_second_batch_keeps_decreases_or_increases_one(self):
        for counts, expected in [([5] * 20, 2), ([1] * 20, 1), ([10] * 20, 3)]:
            with self.subTest(counts=counts[0]):
                policy = grouping.DynamicLoRaGrouping()
                self.complete(policy, [5] * 20)
                self.assertEqual(policy.select(counts).num_groups, expected)

    def test_second_batch_reduction_accepts_equality(self):
        policy = grouping.DynamicLoRaGrouping(100, 20, 0.5)
        self.complete(policy, [5] * 20)
        plan = policy.select([5] * 10)
        self.assertEqual((plan.num_groups, plan.compute), (1, 500))
        # Exactly threshold for g-1, after starting with g=2.
        plan = policy.select([5] * 20 + [0])
        self.assertEqual(plan.num_groups, 2)
        plan = policy.select([10] * 10)
        self.assertEqual((plan.num_groups, plan.compute), (1, 1000))

    def test_later_decrease_uses_previous_two_completed_throughputs(self):
        policy = grouping.DynamicLoRaGrouping()
        self.complete(policy, [5] * 20, elapsed=1)
        self.complete(policy, [5] * 20, elapsed=2)
        plan = self.complete(policy, [5] * 20)
        self.assertEqual((plan.num_groups, plan.reason), (1, "throughput_decreased"))
        # The specified rule does not recheck C(g-1) before decreasing.
        self.assertFalse(policy.last_record["within_budget"])
        self.assertEqual(policy.select([5] * 20).num_groups, 2)

    def test_later_over_budget_takes_precedence_over_falling_throughput(self):
        policy = grouping.DynamicLoRaGrouping()
        self.complete(policy, [5] * 20, 1)
        self.complete(policy, [5] * 20, 2)
        plan = policy.select([10] * 20)
        self.assertEqual((plan.num_groups, plan.reason), (3, "over_budget"))

    def test_later_equal_threshold_keeps_groups_even_when_throughput_falls(self):
        policy = grouping.DynamicLoRaGrouping(100, 20, 0.5)
        self.complete(policy, [5] * 20, 1)
        self.complete(policy, [5] * 20, 2)
        plan = policy.select([5] * 20)
        self.assertEqual((plan.num_groups, plan.compute, plan.reason), (2, 1000, "keep"))

    def test_later_keeps_groups_for_steady_or_rising_throughput(self):
        for elapsed in (1.0, 0.5):
            policy = grouping.DynamicLoRaGrouping()
            self.complete(policy, [5] * 20, 1)
            self.complete(policy, [5] * 20, elapsed)
            self.assertEqual(policy.select([5] * 20).num_groups, 2)

    def test_balancing_is_deterministic_and_ignores_idle_adapters(self):
        self.assertEqual(grouping.balanced_adapter_partitions([8, 4, 3, 2, 0], 2),
                         [[0], [1, 2, 3]])
        plan = grouping.DynamicLoRaGrouping().select([8, 4, 3, 2, 0])
        self.assertEqual(plan.compute, 17 * 4)

    def test_group_boundaries_and_unattainable_budget(self):
        policy = grouping.DynamicLoRaGrouping(1, 1, 0.8)
        plan = self.complete(policy, [2, 2])
        self.assertEqual((plan.num_groups, plan.compute), (2, 4))
        self.assertFalse(policy.last_record["within_budget"])
        self.assertEqual(policy.select([2, 2]).num_groups, 2)
        plan = self.complete(policy, [0, 3])
        self.assertEqual((plan.num_groups, plan.partitions), (1, [[1]]))
        self.assertEqual(policy.select([1, 0]).num_groups, 1)

    def test_record_is_requests_per_elapsed_and_history_is_bounded(self):
        policy = grouping.DynamicLoRaGrouping()
        for _ in range(4):
            self.complete(policy, [3, 2], 0.25)
        self.assertEqual(policy.last_record["throughput"], 20)
        self.assertEqual(policy.last_record["batch_index"], 4)
        self.assertEqual(list(policy.throughputs), [20, 20])
        with self.assertRaises(ValueError):
            policy.finish(policy.select([1]), 0)

    def test_invalid_configuration_and_empty_batch(self):
        for args in [(0, 20, .8), (1, -1, .8), (1.5, 20, .8),
                     (100, 20, 0), (100, 20, 1.1), (100, 20, float("nan"))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                grouping.DynamicLoRaGrouping(*args)
        for counts in ([], [0, 0], [-1, 1], [0.5, 1]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                grouping.DynamicLoRaGrouping().select(counts)


class WorkerGroupingTests(unittest.TestCase):
    def setUp(self):
        self.clock = types.SimpleNamespace(perf_counter=Mock(side_effect=[10.0, 10.25]))
        self.logger = types.SimpleNamespace(info=Mock())
        self.namespace = dict(torch=torch, time=self.clock, wraps=wraps, json=json,
                              ExecType=types.SimpleNamespace(LORA=3, REPLICATED=1, PEFT=2),
                              use_batch_partitions=grouping.use_batch_partitions,
                              phase=lambda name: nullcontext(), logger=self.logger)
        load_nodes("vllm/worker/worker.py", self.namespace, names={"group_lora_batch"})
        self.worker = types.SimpleNamespace(
            exec_type=3, device="cpu", lora_grouping=grouping.DynamicLoRaGrouping(6, 1, 1),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1))
        self.metadata = [types.SimpleNamespace(model_id=i, is_prompt=True,
                                               seq_data={0: "prompt", 1: "beam"})
                         for i in (10, 10, 20, 30)]

    def test_real_execute_reuses_request_partitions_for_all_native_projections(self):
        namespace = ops_namespace()
        torch.manual_seed(37)
        ids = torch.tensor([2] * 11 + [0] * 4 + [1] * 2 + [0] * 7)
        mapping = torch.nn.functional.one_hot(ids, 3).float()
        x, a, b = torch.randn(24, 4), torch.randn(3, 4, 6), torch.randn(3, 6, 7)
        weight = types.SimpleNamespace(active_lora_As=a, active_lora_Bs=b)
        expected = torch.stack([x[i] @ a[j] @ b[j] for i, j in enumerate(ids)])
        projections = []
        for name in ("LoRaColumnParallelLinear", "QKV_LoRaColumnParallelLinear", "LoRaRowParallelLinear"):
            projections.append(load_nodes(
                "vllm/model_executor/parallel_utils/tensor_parallel/layers.py", namespace,
                class_name=name, names={"all_lora"})["all_lora"])

        def model(**kwargs):
            self.assertEqual(grouping.current_batch_partitions(), [[0], [1, 2]])
            # Serving must never rebalance from token counts inside a layer.
            for projection in projections:
                torch.testing.assert_close(projection(None, x, mapping, weight), expected)
            return {0: "output"}

        namespace["_build_balanced_adapter_partitions"] = Mock(side_effect=AssertionError("rebuilt"))
        self.worker.adjust_lora_adapter = Mock(return_value=False)
        self.worker._prepare_inputs = Mock(return_value=("tokens", "positions", "metadata"))
        self.worker.model = model
        self.worker.lora_engine = "weights"
        self.worker.gpu_cache = "cache"
        self.namespace["profile_worker"] = lambda fn: fn
        execute = load_nodes("vllm/worker/worker.py", self.namespace,
                             class_name="Worker", names={"execute_model"})["execute_model"]
        with patch.object(torch.cuda, "synchronize") as sync:
            output = execute(self.worker, self.metadata, [10, 20, 30], [10, 20, 30], {}, {}, {})
        self.assertEqual(output, {0: "output"})
        self.assertEqual(sync.call_count, 2)
        record = self.worker.lora_grouping.last_record
        self.assertEqual((record["num_requests"], record["compute"], record["throughput"]), (4, 6, 16))
        self.assertIsNone(grouping.current_batch_partitions())
        logged = json.loads(self.logger.info.call_args.args[1])
        self.assertEqual(logged, record)

    def test_empty_and_non_lora_batches_do_not_advance_or_synchronize(self):
        fn = Mock(return_value={})
        wrapped = self.namespace["group_lora_batch"](fn)
        with patch.object(torch.cuda, "synchronize") as sync:
            wrapped(self.worker, [], [], [])
            self.worker.exec_type = 1
            wrapped(self.worker, self.metadata, [], [10, 20, 30])
        sync.assert_not_called()
        self.assertEqual(self.worker.lora_grouping.completed_batches, 0)

    def test_exception_resets_context_and_does_not_record_failed_batch(self):
        wrapped = self.namespace["group_lora_batch"](Mock(side_effect=RuntimeError("failed")))
        with patch.object(torch.cuda, "synchronize"), self.assertRaises(RuntimeError):
            wrapped(self.worker, self.metadata, [], [10, 20, 30])
        self.assertIsNone(grouping.current_batch_partitions())
        self.assertEqual(self.worker.lora_grouping.completed_batches, 0)

    def test_parallel_ranks_use_same_max_duration(self):
        self.worker.parallel_config.tensor_parallel_size = 2
        self.namespace["get_model_parallel_group"] = lambda: "model-group"
        self.namespace["dist"] = types.SimpleNamespace(
            all_reduce=Mock(side_effect=lambda duration, **kwargs: duration.fill_(0.5)),
            ReduceOp=types.SimpleNamespace(MAX="max"), get_rank=lambda **kwargs: 0)
        wrapped = self.namespace["group_lora_batch"](Mock(return_value={}))
        with patch.object(torch.cuda, "synchronize"):
            wrapped(self.worker, self.metadata, [], [10, 20, 30])
        self.assertEqual(self.worker.lora_grouping.last_record["throughput"], 8)
        self.namespace["dist"].all_reduce.assert_called_once()

    def test_single_active_adapter_selects_correct_gpu_weight_after_multi_adapter_batch(self):
        namespace = {"torch": torch}
        adjust = load_nodes("vllm/worker/lora_engine.py", namespace,
                            class_name="LoRaWeight", names={"adjust_lora_adapter"})["adjust_lora_adapter"]
        a, b = torch.randn(3, 4, 2), torch.randn(3, 2, 5)
        weight = types.SimpleNamespace(lora_As=a, lora_Bs=b,
                                      lora_engine=types.SimpleNamespace(lora_config=types.SimpleNamespace(gpu_capacity=3)))
        adjust(weight, [0, 1, 2])
        adjust(weight, [2])
        torch.testing.assert_close(weight.active_lora_As, a[2:3])
        torch.testing.assert_close(weight.active_lora_Bs, b[2:3])


class CudaProjectionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_parallel_stream_projections_match_reference(self):
        namespace = ops_namespace()
        ids = torch.tensor([2, 0, 3, 1, 0, 3, 2, 0], device="cuda")
        mapping = torch.nn.functional.one_hot(ids, 4).float()
        x = torch.randn(8, 16, device="cuda")
        a = torch.randn(4, 16, 6, device="cuda")
        b = torch.randn(4, 6, 24, device="cuda")
        expected = torch.stack([x[i] @ a[j] @ b[j] for i, j in enumerate(ids)])
        with grouping.use_batch_partitions([[0], [1, 2], [3]]):
            for _ in range(4):
                actual = namespace["_partitioned_all_lora"](x, mapping, a, b, 16, 8, r_factor=3)
                torch.testing.assert_close(actual, expected)


class ArgumentTests(unittest.TestCase):
    def test_custom_cli_parameters_reach_lora_config(self):
        import dataclasses
        namespace = dict(argparse=argparse, dataclasses=dataclasses, dataclass=dataclasses.dataclass)
        load_nodes("vllm/engine/arg_utils.py", namespace, names={"EngineArgs"})
        args_class = namespace["EngineArgs"]
        parser = args_class.add_cli_args(argparse.ArgumentParser())
        args = args_class.from_cli_args(parser.parse_args([
            "--lora-grouping-reference-requests", "50",
            "--lora-grouping-reference-adapters", "30",
            "--lora-grouping-alpha", "0.6"]))
        for name in ("ModelConfig", "CacheConfig", "ParallelConfig", "SchedulerConfig"):
            namespace[name] = Mock()
        namespace["DynamicLoRaGrouping"] = grouping.DynamicLoRaGrouping
        load_nodes("vllm/config.py", namespace, names={"LoRaConfig"})
        config = args.create_engine_configs()[-1]
        policy = grouping.DynamicLoRaGrouping(config.grouping_reference_requests,
                                              config.grouping_reference_adapters,
                                              config.grouping_alpha)
        self.assertEqual((policy.c_full, policy.threshold), (1500, 900))


if __name__ == "__main__":
    unittest.main()
