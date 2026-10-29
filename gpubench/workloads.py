"""Benchmark workloads.

Three kernels were chosen because they stress hardware in different ways:

* ``matmul``  -- compute-bound: 2*n^3 FLOPs on 3*n^2 elements. Lots of math per
  byte, so a GPU's thousands of cores pay off once ``n`` is large enough.
* ``axpy``    -- memory-bound: 2 FLOPs per 3 elements moved. Speed is limited
  by memory bandwidth, not by math.
* ``mean``    -- a reduction: many values collapse into one, which needs
  cooperation between threads on a GPU.

Every workload knows how to create inputs, run itself on a backend, verify the
result against a float64 reference, and count its FLOPs and bytes so raw times
can be turned into GFLOP/s and GB/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .backends import DTYPE_BYTES, NUMPY_DTYPES, Backend

Inputs = tuple[Any, ...]


def _quantize(array: np.ndarray, dtype: str) -> np.ndarray:
    """Round inputs to the target precision on the host.

    The float64 reference is computed from these *already rounded* values, so
    verification measures arithmetic error only, not input rounding.
    """
    np_dtype = NUMPY_DTYPES.get(dtype, np.float32)
    return array.astype(np_dtype)


@dataclass(frozen=True)
class Workload:
    name: str
    size_label: str  # what "size" means for this workload
    make_inputs: Callable[[np.random.Generator, int, str], Inputs]
    run: Callable[[Backend, Inputs], Any]
    reference: Callable[[Inputs, Any, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    flops: Callable[[int], float]
    bytes_moved: Callable[[int, str], float]


# --- matmul ---------------------------------------------------------------

def _matmul_inputs(rng, n, dtype):
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    return _quantize(a, dtype), _quantize(b, dtype)


def _matmul_run(backend, inputs):
    a, b = inputs
    return backend.matmul(a, b)


def _matmul_reference(host_inputs, out, rng, max_rows=64):
    """Check a random subset of rows: O(rows * n^2) instead of O(n^3).

    A wrong kernel almost never fails on just a handful of rows, and this keeps
    verification of a 4096x4096 product fast on the CPU.
    """
    a, b = (x.astype(np.float64) for x in host_inputs)
    n = a.shape[0]
    rows = rng.choice(n, size=min(n, max_rows), replace=False)
    expected = a[rows] @ b
    return np.asarray(out)[rows], expected


MATMUL = Workload(
    name="matmul",
    size_label="matrix size n (n x n)",
    make_inputs=_matmul_inputs,
    run=_matmul_run,
    reference=_matmul_reference,
    flops=lambda n: 2.0 * n**3,
    bytes_moved=lambda n, dtype: 3.0 * n * n * DTYPE_BYTES[dtype],
)


# --- axpy -----------------------------------------------------------------

AXPY_ALPHA = 1.5


def _axpy_inputs(rng, n, dtype):
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    return _quantize(x, dtype), _quantize(y, dtype)


def _axpy_run(backend, inputs):
    x, y = inputs
    return backend.axpy(AXPY_ALPHA, x, y)


def _axpy_reference(host_inputs, out, rng):
    x, y = (v.astype(np.float64) for v in host_inputs)
    return np.asarray(out), AXPY_ALPHA * x + y


AXPY = Workload(
    name="axpy",
    size_label="vector length n",
    make_inputs=_axpy_inputs,
    run=_axpy_run,
    reference=_axpy_reference,
    flops=lambda n: 2.0 * n,
    bytes_moved=lambda n, dtype: 3.0 * n * DTYPE_BYTES[dtype],
)


# --- mean -----------------------------------------------------------------

def _mean_inputs(rng, n, dtype):
    # Values in [0, 1): the mean is ~0.5, far from zero, which keeps the
    # relative error meaningful and avoids float16 overflow.
    return (_quantize(rng.random(n), dtype),)


def _mean_run(backend, inputs):
    (x,) = inputs
    return backend.mean(x)


def _mean_reference(host_inputs, out, rng):
    (x,) = host_inputs
    return np.atleast_1d(np.asarray(out)), np.atleast_1d(x.astype(np.float64).mean())


MEAN = Workload(
    name="mean",
    size_label="vector length n",
    make_inputs=_mean_inputs,
    run=_mean_run,
    reference=_mean_reference,
    flops=lambda n: float(n),
    bytes_moved=lambda n, dtype: float(n) * DTYPE_BYTES[dtype],
)


WORKLOADS: dict[str, Workload] = {w.name: w for w in (MATMUL, AXPY, MEAN)}


def get_workload(name: str) -> Workload:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ValueError(f"Unknown workload {name!r}. Options: {sorted(WORKLOADS)}") from None
