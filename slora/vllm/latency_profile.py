"""Opt-in synchronized, exclusive wall-time profiling for serving workers."""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import time

_ACTIVE = ContextVar("latency_profile", default=None)


def enabled():
    return os.environ.get("FUSELORA_PROFILE", "0") == "1"


@contextmanager
def phase(name):
    state = _ACTIVE.get()
    if state is None:
        yield
        return
    sync, totals, stack = state
    sync()
    start = time.perf_counter()
    frame = [0.0]
    stack.append(frame)
    try:
        yield
    finally:
        sync()
        elapsed = time.perf_counter() - start
        stack.pop()
        totals[name] += max(0.0, elapsed - frame[0])
        if stack:
            stack[-1][0] += elapsed


def profile_worker(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not enabled():
            return fn(self, *args, **kwargs)
        if self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("FUSELORA_PROFILE requires pipeline_parallel_size=1")
        import torch
        totals = defaultdict(float)
        token = _ACTIVE.set((lambda: torch.cuda.synchronize(self.device), totals, []))
        try:
            with phase("worker_other"):
                output = fn(self, *args, **kwargs)
            # A request experiences the full batch duration, not duration/batch size.
            # Tensor-parallel ranks are replicas of the timing view, never summed.
            for item in output.values():
                item.latency_breakdown = dict(totals)
            return output
        finally:
            _ACTIVE.reset(token)
    return wrapped
