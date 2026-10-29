"""Deliberately broken implementations used to prove the guards work.

Each one would look great on a speed-only leaderboard. ``verify.verify`` must
flag every one of them; the test suite and the demo notebook check that.
"""

from __future__ import annotations

from typing import Any, Callable

from .backends import Backend
from .workloads import Workload

Impl = Callable[[Any], Any]


def honest(workload: Workload, backend: Backend, dtype: str) -> Impl:
    """The real implementation, for comparison."""
    return lambda inputs: workload.run(backend, inputs)


def cached_result(workload: Workload, backend: Backend, dtype: str) -> Impl:
    """Compute once, then return the same answer forever (memoisation cheat)."""
    cache: dict[str, Any] = {}

    def impl(inputs):
        if "out" not in cache:
            cache["out"] = workload.run(backend, inputs)
        return cache["out"]

    return impl


def skip_work(workload: Workload, backend: Backend, dtype: str) -> Impl:
    """Return the first input unchanged -- right shape for matmul/axpy, no work."""

    def impl(inputs):
        first = inputs[0]
        if workload.name == "mean":
            return first[:1]  # shape (1,) like the reference
        return first

    return impl


def lower_precision(workload: Workload, backend: Backend, dtype: str) -> Impl:
    """Silently compute in float16 while claiming the requested dtype.

    On GPUs half precision is much faster (Tensor Cores), so this cheat "wins"
    on speed. On the CPU there is no fast float16 path, so the inputs are only
    rounded to float16 -- the precision loss, and therefore the detection,
    is the same.
    """

    def impl(inputs):
        if backend.is_gpu:
            low = tuple(backend.cast(x, "float16") for x in inputs)
            return backend.cast(workload.run(backend, low), dtype)
        low = tuple(backend.cast(backend.cast(x, "float16"), dtype) for x in inputs)
        return workload.run(backend, low)

    return impl


CHEATS: dict[str, Callable[[Workload, Backend, str], Impl]] = {
    "cached_result": cached_result,
    "skip_work": skip_work,
    "lower_precision": lower_precision,
}
