"""Transfer-fit and memory-prediction maths (no GPU needed)."""

import numpy as np
import pytest

from gpubench.memory import (
    ALLOCATOR_BLOCK,
    MLPSpec,
    predict_matmul_peak,
    predict_mlp_inference_peak,
    tensor_bytes,
)
from gpubench.transfer import fit_latency_bandwidth, fit_table


def test_fit_recovers_known_latency_and_bandwidth():
    sizes = np.array([2**k for k in range(10, 29)], dtype=float)
    latency, bandwidth = 8e-6, 12e9
    times = latency + sizes / bandwidth
    fit = fit_latency_bandwidth(sizes, times)
    assert fit.latency_s == pytest.approx(latency, rel=1e-6)
    assert fit.bandwidth_gb_per_s == pytest.approx(12.0, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)


def test_fit_is_robust_to_noise():
    rng = np.random.default_rng(0)
    sizes = np.array([2**k for k in range(10, 29)], dtype=float)
    times = (10e-6 + sizes / 6e9) * rng.uniform(0.95, 1.05, sizes.size)
    fit = fit_latency_bandwidth(sizes, times)
    assert 5 < fit.bandwidth_gb_per_s < 7
    assert 5e-6 < fit.latency_s < 15e-6


def test_fit_table_groups_by_direction_and_memory():
    rows = []
    for direction in ("H2D", "D2H"):
        for memory, bw in (("pinned", 12e9), ("pageable", 4e9)):
            for k in range(12, 24):
                b = 2**k
                rows.append({"direction": direction, "memory": memory, "bytes": b,
                             "time_median": 5e-6 + b / bw})
    table = {(r["direction"], r["memory"]): r for r in fit_table(rows)}
    assert len(table) == 4
    assert table[("H2D", "pinned")]["bandwidth_gb_per_s"] == pytest.approx(12, rel=1e-3)
    assert table[("D2H", "pageable")]["bandwidth_gb_per_s"] == pytest.approx(4, rel=1e-3)


def test_tensor_bytes_rounds_to_allocator_blocks():
    assert tensor_bytes(1, "float32") == ALLOCATOR_BLOCK
    assert tensor_bytes(128, "float32") == 512
    assert tensor_bytes(129, "float32") == 1024


def test_matmul_peak_is_three_matrices():
    assert predict_matmul_peak(1024, "float32") == 3 * 1024 * 1024 * 4
    assert predict_matmul_peak(1024, "float16") == predict_matmul_peak(1024, "float32") // 2


def test_mlp_layers_and_params():
    spec = MLPSpec(widths=(4, 8, 2), batch=3)
    assert spec.layers() == [("linear", 4, 8), ("relu", 8, 8), ("linear", 8, 2)]
    assert spec.n_params() == 4 * 8 + 8 + 8 * 2 + 2


def test_mlp_peak_by_hand():
    # Large enough that 512-byte rounding does not matter.
    spec = MLPSpec(widths=(1024, 4096, 1024), batch=64)
    pred = predict_mlp_inference_peak(spec, "float32")
    weights = (1024 * 4096 + 4096 + 4096 * 1024 + 1024) * 4
    assert pred["weights"] == pytest.approx(weights, abs=4 * ALLOCATOR_BLOCK)
    # Widest moment: caller's x (64x1024) + ReLU input and output (2 x 64x4096).
    acts = (64 * 1024 + 2 * 64 * 4096) * 4
    assert pred["activations"] == acts
    assert pred["total"] == pred["weights"] + pred["activations"]


def test_float16_halves_memory():
    spec = MLPSpec(widths=(2048, 8192, 2048), batch=128)
    full = predict_mlp_inference_peak(spec, "float32")["total"]
    half = predict_mlp_inference_peak(spec, "float16")["total"]
    assert half == pytest.approx(full / 2, rel=0.01)
