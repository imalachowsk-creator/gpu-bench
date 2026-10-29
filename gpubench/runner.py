"""Run a benchmark suite and save raw results.

Output layout (one directory per run)::

    results/<run>/
        env.json        hardware and library versions
        compute.csv     one row per (workload, backend, dtype, size)
        transfer.csv    host<->device copies           (GPU only)
        memory.csv      predicted vs measured memory   (GPU only)
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import cheats
from .backends import Backend, available_backends, make_backend
from .timing import measure, summarize
from .verify import PhysicalLimits, verify
from .workloads import Workload, get_workload

PRESETS: dict[str, dict[str, list[int]]] = {
    # Small enough for CI and laptops; still shows the trends.
    "quick": {
        "matmul": [32, 64, 128, 256, 512],
        "axpy": [2**k for k in range(10, 21, 2)],
        "mean": [2**k for k in range(10, 21, 2)],
    },
    # For a real GPU (fits easily in a 16 GB Colab T4).
    "full": {
        "matmul": [32, 64, 128, 256, 512, 1024, 2048, 4096],
        "axpy": [2**k for k in range(10, 27, 2)],
        "mean": [2**k for k in range(10, 27, 2)],
    },
}


@dataclass
class SuiteConfig:
    preset: str = "quick"
    workloads: list[str] = field(default_factory=lambda: ["matmul", "axpy", "mean"])
    backends: list[str] | None = None  # None = everything available
    dtypes: list[str] = field(default_factory=lambda: ["float32", "float16"])
    repeats: int = 30
    warmup: int = 3
    max_seconds: float = 2.0
    seed: int = 0
    skip_transfer: bool = False
    skip_memory: bool = False


def environment() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gib"] = round(props.total_memory / 2**30, 1)
            info["gpu_sm_count"] = props.multi_processor_count
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


def run_case(
    workload: Workload,
    backend: Backend,
    size: int,
    dtype: str,
    *,
    impl_factory: Callable = cheats.honest,
    repeats: int = 30,
    warmup: int = 3,
    max_seconds: float = 2.0,
    seed: int = 0,
    limits: PhysicalLimits = PhysicalLimits(),
    implementation: str = "honest",
) -> dict:
    """Time one configuration, then verify it. Returns one result row."""
    rng = np.random.default_rng(seed)
    host_inputs = workload.make_inputs(rng, size, dtype)
    device_inputs = tuple(backend.from_numpy(x, dtype) for x in host_inputs)
    backend.sync()
    impl = impl_factory(workload, backend, dtype)

    samples = measure(
        lambda: impl(device_inputs),
        backend.sync,
        warmup=warmup,
        repeats=repeats,
        max_seconds=max_seconds,
    )
    stats = summarize(samples)
    verdict = verify(
        workload, backend, impl, size, dtype,
        seed=seed, timed_inputs=host_inputs, median_seconds=stats.median, limits=limits,
    )
    del device_inputs

    return {
        "workload": workload.name,
        "backend": backend.name,
        "device": "gpu" if backend.is_gpu else "cpu",
        "implementation": implementation,
        "dtype": dtype,
        "size": size,
        **stats.as_dict("time_"),
        "gflops": workload.flops(size) / stats.median / 1e9,
        "gb_per_s": workload.bytes_moved(size, dtype) / stats.median / 1e9,
        "status": verdict.status,
        "rel_error": verdict.error,
        "reasons": "; ".join(verdict.reasons),
    }


def run_compute_suite(cfg: SuiteConfig, log: Callable[[str], None] = print) -> pd.DataFrame:
    sizes = PRESETS[cfg.preset]
    backend_names = cfg.backends or available_backends()
    rows = []
    for bname in backend_names:
        backend = make_backend(bname)
        dtypes = [d for d in cfg.dtypes if d in backend.supported_dtypes]
        for wname in cfg.workloads:
            workload = get_workload(wname)
            for dtype in dtypes:
                for size in sizes[wname]:
                    row = run_case(
                        workload, backend, size, dtype,
                        repeats=cfg.repeats, warmup=cfg.warmup,
                        max_seconds=cfg.max_seconds, seed=cfg.seed,
                    )
                    rows.append(row)
                    flag = "" if row["status"] == "ok" else f"  <-- {row['status']}"
                    log(
                        f"{bname:>10} {wname:>6} {dtype:>7} size={size:<9} "
                        f"median={row['time_median'] * 1e3:9.4f} ms  "
                        f"{row['gflops']:9.2f} GFLOP/s{flag}"
                    )
    return pd.DataFrame(rows)


def run_suite(cfg: SuiteConfig, out_dir: str | Path, log: Callable[[str], None] = print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    log(f"Environment: {env.get('gpu') or env['processor']}")

    compute = run_compute_suite(cfg, log)
    compute.to_csv(out / "compute.csv", index=False)

    if env.get("cuda_available"):
        if not cfg.skip_transfer:
            from .transfer import benchmark_transfers

            log("Measuring host<->device transfers ...")
            pd.DataFrame(benchmark_transfers()).to_csv(out / "transfer.csv", index=False)
        if not cfg.skip_memory:
            from .memory import benchmark_memory

            log("Measuring peak memory vs prediction ...")
            dtypes = [d for d in cfg.dtypes if d in ("float32", "float16")]
            pd.DataFrame(benchmark_memory(dtypes)).to_csv(out / "memory.csv", index=False)
    else:
        log("No CUDA GPU: transfer and memory benchmarks skipped.")

    log(f"Raw results written to {out}/")
    return out
