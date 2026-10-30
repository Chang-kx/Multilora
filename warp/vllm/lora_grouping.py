"""Request-based LoRA grouping policy, independent of CUDA and model layers."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple


_BATCH_PARTITIONS = ContextVar("lora_batch_partitions", default=None)


def current_batch_partitions():
    return _BATCH_PARTITIONS.get()


@contextmanager
def use_batch_partitions(partitions):
    token = _BATCH_PARTITIONS.set(partitions)
    try:
        yield
    finally:
        _BATCH_PARTITIONS.reset(token)


def balanced_adapter_partitions(request_counts: Sequence[int],
                                num_groups: int) -> List[List[int]]:
    """Assign whole adapters in descending request load; omit idle adapters."""
    adapter_ids = sorted((i for i, count in enumerate(request_counts) if count > 0),
                         key=lambda i: (-request_counts[i], i))
    if not adapter_ids:
        return []
    num_groups = max(1, min(num_groups, len(adapter_ids)))
    partitions = [[] for _ in range(num_groups)]
    loads = [0] * num_groups
    for adapter_id in adapter_ids:
        target = min(range(num_groups),
                     key=lambda i: (loads[i], len(partitions[i]), i))
        partitions[target].append(adapter_id)
        loads[target] += request_counts[adapter_id]
    return partitions


@dataclass(frozen=True)
class GroupingPlan:
    batch_index: int
    request_counts: Tuple[int, ...]
    previous_groups: int
    partitions: List[List[int]]
    reused_compute: int
    compute: int
    threshold: float
    reason: str

    @property
    def num_groups(self):
        return len(self.partitions)

    @property
    def num_requests(self):
        return sum(self.request_counts)


class DynamicLoRaGrouping:
    def __init__(self, reference_requests: int = 100,
                 reference_adapters: int = 20, alpha: float = 0.8):
        if (isinstance(reference_requests, bool)
                or not isinstance(reference_requests, int)
                or reference_requests <= 0):
            raise ValueError("reference_requests must be a positive integer")
        if (isinstance(reference_adapters, bool)
                or not isinstance(reference_adapters, int)
                or reference_adapters <= 0):
            raise ValueError("reference_adapters must be a positive integer")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1]")
        self.c_full = reference_requests * reference_adapters
        self.threshold = alpha * self.c_full
        self.num_groups = 1
        self.completed_batches = 0
        self.throughputs = deque(maxlen=2)
        self.last_record = None

    def select(self, request_counts: Sequence[int]) -> GroupingPlan:
        counts = tuple(request_counts)
        if any(isinstance(c, bool) or not isinstance(c, int) or c < 0
               for c in counts):
            raise ValueError("request counts must be nonnegative integers")
        max_groups = sum(c > 0 for c in counts)
        if max_groups == 0:
            raise ValueError("cannot select groups for an empty batch")
        cache = {}

        def evaluate(groups):
            if groups not in cache:
                partitions = balanced_adapter_partitions(counts, groups)
                cost = sum(sum(counts[i] for i in part) * len(part)
                           for part in partitions)
                cache[groups] = partitions, cost
            return cache[groups]

        groups = min(self.num_groups, max_groups)
        reused_compute = evaluate(groups)[1]
        reason = "keep"
        if self.completed_batches == 0:
            reason = "initial_search"
            for groups in range(1, max_groups + 1):
                if evaluate(groups)[1] <= self.threshold:
                    break
        elif reused_compute > self.threshold:
            groups = min(groups + 1, max_groups)
            reason = "over_budget"
        elif self.completed_batches == 1:
            if groups > 1 and evaluate(groups - 1)[1] <= self.threshold:
                groups -= 1
                reason = "previous_group_fits"
        elif (reused_compute < self.threshold and groups > 1
              and self.throughputs[-1] < self.throughputs[-2]):
            groups -= 1
            reason = "throughput_decreased"

        partitions, cost = evaluate(groups)
        return GroupingPlan(self.completed_batches + 1, counts, self.num_groups,
                            partitions, reused_compute, cost, self.threshold,
                            reason)

    def finish(self, plan: GroupingPlan, elapsed: float):
        if plan.batch_index != self.completed_batches + 1:
            raise ValueError("batch plans must finish in order and only once")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("batch duration must be finite and positive")
        throughput = plan.num_requests / elapsed
        record = dict(batch_index=plan.batch_index,
                      num_requests=plan.num_requests,
                      num_adapters=sum(c > 0 for c in plan.request_counts),
                      previous_groups=plan.previous_groups,
                      num_groups=plan.num_groups,
                      c_full=self.c_full, threshold=plan.threshold,
                      reused_compute=plan.reused_compute, compute=plan.compute,
                      within_budget=plan.compute <= plan.threshold,
                      elapsed_sec=elapsed, throughput=throughput,
                      previous_throughputs=list(self.throughputs),
                      reason=plan.reason, partitions=plan.partitions)
        self.num_groups = plan.num_groups
        self.completed_batches += 1
        self.throughputs.append(throughput)
        self.last_record = record
        return record
