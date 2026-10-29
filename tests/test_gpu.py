"""GPU-only tests. Skipped automatically when no CUDA device is present
(e.g. in CI); run them on Colab with ``pytest tests/test_gpu.py -v``."""

import pytest

from gpubench.backends import cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="needs a CUDA GPU")


def test_cuda_backend_passes_verification():
    from gpubench.backends import make_backend
    from gpubench.runner import run_case
    from gpubench.workloads import WORKLOADS

    be = make_backend("torch-cuda")
    for workload in WORKLOADS.values():
        size = 512 if workload.name == "matmul" else 1 << 20
        for dtype in ("float32", "float16"):
            row = run_case(workload, be, size, dtype, repeats=5)
            assert row["status"] == "ok", (workload.name, dtype, row["reasons"])


def test_cheats_are_caught_on_gpu():
    from gpubench.demo import cheat_demo, cheat_demo_passed

    df = cheat_demo(backend_name="torch-cuda", size=1024)
    assert cheat_demo_passed(df), df[["implementation", "status", "reasons"]].to_string()


def test_missing_sync_underestimates_time():
    from gpubench.demo import sync_pitfall_demo

    df = sync_pitfall_demo(sizes=(4096,))
    assert df["apparent_speedup"].iloc[0] > 2


def test_transfer_benchmark_runs():
    from gpubench.transfer import benchmark_transfers, fit_table

    rows = benchmark_transfers([2**k for k in range(16, 25, 2)], repeats=5)
    fits = fit_table(rows)
    assert len(fits) == 4
    assert all(f["bandwidth_gb_per_s"] > 0.5 for f in fits)


def test_matmul_memory_prediction_is_close():
    from gpubench.memory import measure_matmul_peak, predict_matmul_peak

    n = 4096
    measured = measure_matmul_peak(n, "float32")
    predicted = predict_matmul_peak(n, "float32")
    assert abs(predicted - measured) / measured < 0.25
