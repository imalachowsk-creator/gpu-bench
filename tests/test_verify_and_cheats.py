import numpy as np
import pytest

from gpubench import cheats
from gpubench.backends import NumpyBackend
from gpubench.runner import run_case
from gpubench.verify import (
    IMPLAUSIBLE,
    INPUT_INDEPENDENT,
    OK,
    WRONG_RESULT,
    PhysicalLimits,
    normalized_error,
    verify,
)
from gpubench.workloads import AXPY, MATMUL, MEAN, WORKLOADS

BACKEND = NumpyBackend()


def test_normalized_error_basics():
    assert normalized_error(np.ones(3), np.ones(3)) == 0.0
    assert normalized_error(np.ones(3), np.ones(4)) == float("inf")
    assert normalized_error(np.array([np.nan]), np.array([1.0])) == float("inf")
    assert normalized_error(np.array([1.1]), np.array([1.0])) == pytest.approx(0.1)


@pytest.mark.parametrize("workload", list(WORKLOADS.values()), ids=list(WORKLOADS))
def test_honest_implementation_passes(workload):
    size = 128 if workload is MATMUL else 10_000
    impl = cheats.honest(workload, BACKEND, "float32")
    verdict = verify(workload, BACKEND, impl, size, "float32")
    assert verdict.status == OK, verdict.reasons


def test_cached_result_is_caught_as_input_independent():
    row = run_case(MATMUL, BACKEND, 128, "float32", impl_factory=cheats.cached_result, repeats=5)
    assert row["status"] == INPUT_INDEPENDENT


@pytest.mark.parametrize("workload", [MATMUL, AXPY, MEAN], ids=["matmul", "axpy", "mean"])
def test_skip_work_is_caught(workload):
    size = 128 if workload is MATMUL else 10_000
    row = run_case(workload, BACKEND, size, "float32", impl_factory=cheats.skip_work, repeats=5)
    assert row["status"] != OK


def test_lower_precision_is_caught():
    row = run_case(MATMUL, BACKEND, 256, "float32", impl_factory=cheats.lower_precision, repeats=5)
    assert row["status"] == WRONG_RESULT


def test_impossible_speed_is_flagged():
    impl = cheats.honest(MATMUL, BACKEND, "float32")
    # Claim a 1024^3 matmul took one nanosecond.
    verdict = verify(MATMUL, BACKEND, impl, 1024, "float32", median_seconds=1e-9)
    assert verdict.status == IMPLAUSIBLE
    assert any("TFLOP/s" in r for r in verdict.reasons)


def test_realistic_speed_is_not_flagged():
    impl = cheats.honest(MATMUL, BACKEND, "float32")
    verdict = verify(MATMUL, BACKEND, impl, 256, "float32", median_seconds=1e-3,
                     limits=PhysicalLimits())
    assert verdict.status == OK


def test_crashing_implementation_is_reported_not_raised():
    def boom(inputs):
        raise RuntimeError("kernel launch failed")

    verdict = verify(MATMUL, BACKEND, boom, 64, "float32")
    assert verdict.status == "error"
    assert "kernel launch failed" in verdict.reasons[0]
