"""Predicting peak GPU memory, then checking the prediction.

Running out of GPU memory is the most common way a model "doesn't fit".
Knowing the peak *before* launching a job saves a lot of trial and error,
and halving the bytes per value (float32 -> float16) is the simplest form of
model compression.

The model here is deliberately simple and explicit:

* every tensor costs ``elements * bytes_per_value``, rounded up to the
  512-byte blocks PyTorch's caching allocator hands out;
* for inference through ``nn.Sequential`` we track which tensors are alive at
  each step (weights, the caller's input, the current layer's input and
  output) and take the maximum.

Comparing prediction with ``torch.cuda.max_memory_allocated`` shows what the
simple model leaves out (e.g. library workspaces, allocator rounding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .backends import DTYPE_BYTES

ALLOCATOR_BLOCK = 512  # bytes; PyTorch rounds every allocation up to this


def tensor_bytes(elements: int, dtype: str) -> int:
    raw = elements * DTYPE_BYTES[dtype]
    return int(math.ceil(raw / ALLOCATOR_BLOCK) * ALLOCATOR_BLOCK)


def predict_matmul_peak(n: int, dtype: str) -> int:
    """A, B and C = A @ B are alive at the same time."""
    return 3 * tensor_bytes(n * n, dtype)


@dataclass(frozen=True)
class MLPSpec:
    """A plain MLP: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear."""

    widths: tuple[int, ...]  # e.g. (4096, 8192, 8192, 4096)
    batch: int

    def layers(self) -> list[tuple[str, int, int]]:
        out = []
        for i, (d_in, d_out) in enumerate(zip(self.widths[:-1], self.widths[1:])):
            out.append(("linear", d_in, d_out))
            if i < len(self.widths) - 2:
                out.append(("relu", d_out, d_out))
        return out

    def n_params(self) -> int:
        return sum(d_in * d_out + d_out for d_in, d_out in zip(self.widths[:-1], self.widths[1:]))


def predict_mlp_inference_peak(spec: MLPSpec, dtype: str) -> dict[str, int]:
    """Peak bytes for one forward pass without autograd.

    ``nn.Sequential`` does ``x = layer(x)``: while a layer runs, its input and
    output are both alive; afterwards the input is freed -- unless it is the
    caller's original tensor, which the caller still holds.
    """
    weights = sum(
        tensor_bytes(d_in * d_out, dtype) + tensor_bytes(d_out, dtype)
        for d_in, d_out in zip(spec.widths[:-1], spec.widths[1:])
    )
    caller_input = tensor_bytes(spec.batch * spec.widths[0], dtype)

    peak_activations = 0
    for i, (_, d_in, d_out) in enumerate(spec.layers()):
        current_in = 0 if i == 0 else tensor_bytes(spec.batch * d_in, dtype)
        current_out = tensor_bytes(spec.batch * d_out, dtype)
        peak_activations = max(peak_activations, caller_input + current_in + current_out)

    return {
        "weights": weights,
        "activations": peak_activations,
        "total": weights + peak_activations,
    }


# --- measurement (needs CUDA) ---------------------------------------------

def _torch_dtype(torch, dtype: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


def measure_matmul_peak(n: int, dtype: str, device: str = "cuda") -> int:
    import torch

    dev = torch.device(device)
    tdt = _torch_dtype(torch, dtype)
    # Warm-up so one-time cuBLAS workspace allocations are already in the
    # baseline and are not attributed to this matmul.
    w = torch.ones(64, 64, device=dev, dtype=tdt)
    _ = w @ w
    del w, _
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    a = torch.randn(n, n, device=dev, dtype=tdt)
    b = torch.randn(n, n, device=dev, dtype=tdt)
    c = a @ b
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) - base
    del a, b, c
    torch.cuda.empty_cache()
    return int(peak)


def measure_mlp_inference_peak(spec: MLPSpec, dtype: str, device: str = "cuda") -> int:
    import torch
    from torch import nn

    dev = torch.device(device)
    tdt = _torch_dtype(torch, dtype)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    modules: list[nn.Module] = []
    for kind, d_in, d_out in spec.layers():
        modules.append(nn.Linear(d_in, d_out) if kind == "linear" else nn.ReLU())
    model = nn.Sequential(*modules).to(device=dev, dtype=tdt)
    x = torch.randn(spec.batch, spec.widths[0], device=dev, dtype=tdt)
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) - base
    del model, x, y
    torch.cuda.empty_cache()
    return int(peak)


DEFAULT_MATMUL_SIZES = [1024, 2048, 4096, 8192]
DEFAULT_MLP_SPECS = [
    MLPSpec(widths=(1024, 4096, 4096, 1024), batch=64),
    MLPSpec(widths=(4096, 8192, 8192, 4096), batch=256),
    MLPSpec(widths=(4096, 16384, 4096), batch=1024),
]


def benchmark_memory(dtypes=("float32", "float16"), device: str = "cuda") -> list[dict]:
    rows = []
    for dtype in dtypes:
        for n in DEFAULT_MATMUL_SIZES:
            predicted = predict_matmul_peak(n, dtype)
            measured = measure_matmul_peak(n, dtype, device)
            rows.append(_row("matmul", f"n={n}", dtype, predicted, measured))
        for spec in DEFAULT_MLP_SPECS:
            predicted = predict_mlp_inference_peak(spec, dtype)["total"]
            measured = measure_mlp_inference_peak(spec, dtype, device)
            label = f"MLP {'-'.join(map(str, spec.widths))} b={spec.batch}"
            rows.append(_row("mlp", label, dtype, predicted, measured))
    return rows


def _row(kind, label, dtype, predicted, measured) -> dict:
    return {
        "kind": kind,
        "case": label,
        "dtype": dtype,
        "predicted_mib": predicted / 2**20,
        "measured_mib": measured / 2**20,
        "error_pct": 100.0 * (predicted - measured) / measured if measured else float("nan"),
    }
