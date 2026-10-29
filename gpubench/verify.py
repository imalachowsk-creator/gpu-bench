"""Correctness checks and "anti-cheating" guards.

A benchmark that only measures speed rewards implementations that are fast
because they are *wrong*: returning a cached answer, skipping work, or quietly
computing in lower precision. Every timed result therefore has to pass three
independent checks before it is trusted:

1. **Correctness** -- output matches a float64 reference within a tolerance
   that depends on the dtype.
2. **Input dependence** -- the implementation is run again on *fresh* random
   inputs and must still be correct. This catches cached / memoised results.
3. **Physical plausibility** -- the achieved FLOP/s and bytes/s must be below
   what any existing hardware can do. A "1000 PFLOP/s" matmul is a bug.

The same ideas apply to evaluating AI agents: a passing test is not proof that
the task was solved, so the evaluation must be hard to game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .backends import Backend
from .workloads import Workload

# Allowed normalised error per dtype. float16 has ~3 significant decimal
# digits, so a much looser bound is needed than for float32.
TOLERANCES: dict[str, float] = {
    "float64": 1e-10,
    "float32": 1e-4,
    "bfloat16": 2e-2,
    "float16": 1e-2,
}

OK = "ok"
WRONG_RESULT = "wrong_result"
INPUT_INDEPENDENT = "input_independent"
IMPLAUSIBLE = "implausible"
ERROR = "error"


@dataclass(frozen=True)
class PhysicalLimits:
    """Generous upper bounds no current single device exceeds.

    They are not meant to be tight -- only to flag results that are
    impossible, which almost always means the work was not really done.
    """

    max_flops_per_s: float = 5e15  # 5 PFLOP/s
    max_bytes_per_s: float = 2e13  # 20 TB/s


@dataclass
class Verdict:
    status: str = OK
    error: float = float("nan")
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def flag(self, status: str, reason: str) -> None:
        # Keep the first (most important) status but record every reason.
        if self.status == OK:
            self.status = status
        self.reasons.append(reason)


def normalized_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """max |actual - expected| / max |expected| (inf if shapes/values break)."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return float("inf")
    if not np.all(np.isfinite(actual)):
        return float("inf")
    scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(actual - expected)) / scale)


Impl = Callable[[Any], Any]  # device inputs -> device output


def check_once(
    workload: Workload,
    backend: Backend,
    impl: Impl,
    size: int,
    dtype: str,
    rng: np.random.Generator,
    host_inputs: tuple | None = None,
) -> float:
    """Run ``impl`` once and return the normalised error.

    Uses ``host_inputs`` if given, otherwise draws new random inputs.
    """
    if host_inputs is None:
        host_inputs = workload.make_inputs(rng, size, dtype)
    device_inputs = tuple(backend.from_numpy(x, dtype) for x in host_inputs)
    out = impl(device_inputs)
    backend.sync()
    actual, expected = workload.reference(host_inputs, backend.to_numpy(out), rng)
    return normalized_error(actual, expected)


def verify(
    workload: Workload,
    backend: Backend,
    impl: Impl,
    size: int,
    dtype: str,
    *,
    seed: int = 0,
    timed_inputs: tuple | None = None,
    median_seconds: float | None = None,
    limits: PhysicalLimits = PhysicalLimits(),
) -> Verdict:
    """Run all checks.

    ``timed_inputs`` should be the host inputs that were used while timing:
    the first check proves the timed computation was right, the second (on
    fresh inputs) proves the answer actually depends on the input.
    """
    verdict = Verdict()
    tol = TOLERANCES[dtype]
    rng = np.random.default_rng(seed + 1_000_003)  # different from timing inputs

    try:
        err_first = check_once(workload, backend, impl, size, dtype, rng, timed_inputs)
        err_fresh = check_once(workload, backend, impl, size, dtype, rng)
    except Exception as exc:  # a crash is a failed check, not a crashed suite
        verdict.flag(ERROR, f"{type(exc).__name__}: {exc}")
        return verdict

    verdict.error = max(err_first, err_fresh)
    if err_first > tol:
        verdict.flag(WRONG_RESULT, f"error {err_first:.2e} > tolerance {tol:.0e} for {dtype}")
    elif err_fresh > tol:
        verdict.flag(
            INPUT_INDEPENDENT,
            f"correct on the first inputs but error {err_fresh:.2e} on fresh inputs "
            "(cached or stale result?)",
        )

    if median_seconds is not None and median_seconds > 0:
        flops_rate = workload.flops(size) / median_seconds
        bytes_rate = workload.bytes_moved(size, dtype) / median_seconds
        if flops_rate > limits.max_flops_per_s:
            verdict.flag(IMPLAUSIBLE, f"{flops_rate / 1e12:.0f} TFLOP/s exceeds any real device")
        if bytes_rate > limits.max_bytes_per_s:
            verdict.flag(IMPLAUSIBLE, f"{bytes_rate / 1e12:.0f} TB/s exceeds any real device")
    return verdict
