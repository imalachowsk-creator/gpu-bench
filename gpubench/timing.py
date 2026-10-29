"""Timing and summary statistics.

Rules this module enforces:

1. **Warm-up.** First calls pay one-off costs (CUDA context creation, kernel
   selection, memory allocation, cold caches). They are run and discarded.
2. **Synchronise around every timed call.** Otherwise asynchronous GPU work
   escapes the measurement window.
3. **Repeat and report robust statistics.** Single timings are noisy and
   skewed by occasional slow runs, so the median and IQR are reported rather
   than a single number or the mean.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


def _no_sync() -> None:
    return None


def measure(
    fn: Callable[[], object],
    sync: Callable[[], None] = _no_sync,
    *,
    warmup: int = 3,
    repeats: int = 30,
    min_repeats: int = 5,
    max_seconds: float = 2.0,
) -> list[float]:
    """Return a list of wall-clock durations (seconds) of ``fn()``.

    The loop stops early once ``min_repeats`` samples exist and more than
    ``max_seconds`` have been spent, so huge problem sizes do not take forever.
    """
    if repeats < 1 or min_repeats < 1:
        raise ValueError("repeats and min_repeats must be >= 1")

    for _ in range(warmup):
        fn()
    sync()

    samples: list[float] = []
    budget_start = time.perf_counter()
    for i in range(repeats):
        sync()  # make sure nothing from before leaks into this sample
        t0 = time.perf_counter()
        fn()
        sync()  # wait until the work has actually finished
        samples.append(time.perf_counter() - t0)
        spent = time.perf_counter() - budget_start
        if i + 1 >= min_repeats and spent > max_seconds:
            break
    return samples


@dataclass(frozen=True)
class Stats:
    n: int
    median: float
    q1: float
    q3: float
    iqr: float
    mean: float
    std: float
    min: float
    max: float
    n_outliers: int  # by the 1.5 * IQR rule

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def summarize(samples: list[float] | np.ndarray) -> Stats:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no samples to summarize")
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.sum((x < lo) | (x > hi)))
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return Stats(
        n=int(x.size),
        median=float(med),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        mean=float(x.mean()),
        std=std,
        min=float(x.min()),
        max=float(x.max()),
        n_outliers=n_out,
    )
