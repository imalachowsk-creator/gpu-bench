"""Host <-> device transfer benchmark and a latency/bandwidth model.

Copying data over PCIe is often what makes "GPU is faster" untrue for small
problems. The copy time is well described by a straight line::

    time(bytes) = latency + bytes / bandwidth

so an ordinary least-squares fit of measured times against size gives two
physically meaningful numbers: the fixed per-copy overhead (intercept) and
the effective bandwidth (1 / slope).

Two kinds of host memory are compared:

* *pageable* -- normal memory; the driver must first copy it into a pinned
  staging buffer, which costs extra time.
* *pinned* (page-locked) -- the GPU's DMA engine can read it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timing import measure, summarize


@dataclass(frozen=True)
class LinearFit:
    latency_s: float  # intercept
    bandwidth_bytes_per_s: float  # 1 / slope
    r_squared: float

    @property
    def bandwidth_gb_per_s(self) -> float:
        return self.bandwidth_bytes_per_s / 1e9


def fit_latency_bandwidth(sizes_bytes, times_s) -> LinearFit:
    """Weighted least-squares fit of ``time = latency + bytes / bandwidth``.

    Sizes span five orders of magnitude, so a plain fit would be dominated by
    the largest copies and the intercept (latency) would be meaningless.
    Weighting each point by ``1 / time`` minimises *relative* error instead,
    so small and large copies count equally.
    """
    x = np.asarray(sizes_bytes, dtype=np.float64)
    y = np.asarray(times_s, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two points to fit a line")
    slope, intercept = np.polyfit(x, y, deg=1, w=1.0 / y)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    bandwidth = 1.0 / slope if slope > 0 else float("inf")
    return LinearFit(latency_s=float(intercept), bandwidth_bytes_per_s=float(bandwidth), r_squared=r2)


DEFAULT_SIZES = [2**k for k in range(12, 29, 2)]  # 4 KiB ... 256 MiB


def benchmark_transfers(sizes_bytes=DEFAULT_SIZES, *, repeats: int = 20, device: str = "cuda") -> list[dict]:
    """Measure H2D and D2H copies for pageable and pinned host memory.

    Requires PyTorch with CUDA. Returns one row per (direction, memory, size).
    """
    import torch

    dev = torch.device(device)
    sync = lambda: torch.cuda.synchronize(dev)  # noqa: E731
    rows = []
    for size in sizes_bytes:
        d_buf = torch.empty(size, dtype=torch.uint8, device=dev)
        for memory in ("pageable", "pinned"):
            h_buf = torch.empty(size, dtype=torch.uint8)
            if memory == "pinned":
                h_buf = h_buf.pin_memory()
            for direction, fn in (
                ("H2D", lambda: d_buf.copy_(h_buf)),
                ("D2H", lambda: h_buf.copy_(d_buf)),
            ):
                samples = measure(fn, sync, warmup=2, repeats=repeats, max_seconds=1.0)
                stats = summarize(samples)
                rows.append(
                    {
                        "direction": direction,
                        "memory": memory,
                        "bytes": size,
                        **stats.as_dict("time_"),
                        "gb_per_s": size / stats.median / 1e9,
                    }
                )
            del h_buf
        del d_buf
    return rows


def fit_table(rows: list[dict]) -> list[dict]:
    """Fit the latency/bandwidth model separately for every direction x memory."""
    out = []
    keys = sorted({(r["direction"], r["memory"]) for r in rows})
    for direction, memory in keys:
        sel = [r for r in rows if r["direction"] == direction and r["memory"] == memory]
        fit = fit_latency_bandwidth([r["bytes"] for r in sel], [r["time_median"] for r in sel])
        out.append(
            {
                "direction": direction,
                "memory": memory,
                "latency_us": fit.latency_s * 1e6,
                "bandwidth_gb_per_s": fit.bandwidth_gb_per_s,
                "r_squared": fit.r_squared,
            }
        )
    return out
