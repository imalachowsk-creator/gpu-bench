"""Small demonstrations used by the CLI and the Colab notebook."""

from __future__ import annotations

import pandas as pd

from . import cheats
from .backends import available_backends, make_backend
from .runner import run_case
from .timing import measure, summarize
from .workloads import MATMUL


def _default_backend() -> str:
    return available_backends()[-1]  # torch-cuda if present, else the best CPU one


def cheat_demo(backend_name: str | None = None, size: int = 1024, dtype: str = "float32") -> pd.DataFrame:
    """Time the honest matmul and every cheat; the guards must flag the cheats."""
    backend = make_backend(backend_name or _default_backend())
    rows = []
    for name, factory in {"honest": cheats.honest, **cheats.CHEATS}.items():
        rows.append(
            run_case(MATMUL, backend, size, dtype, impl_factory=factory,
                     implementation=name, repeats=15, max_seconds=1.0)
        )
    return pd.DataFrame(rows)


def cheat_demo_passed(df: pd.DataFrame) -> bool:
    honest_ok = (df.loc[df["implementation"] == "honest", "status"] == "ok").all()
    cheats_caught = (df.loc[df["implementation"] != "honest", "status"] != "ok").all()
    return bool(honest_ok and cheats_caught)


def sync_pitfall_demo(sizes=(256, 1024, 4096), dtype: str = "float32") -> pd.DataFrame:
    """Time GPU matmul with and without synchronisation.

    Without ``torch.cuda.synchronize()`` the timer stops as soon as the kernel
    is *queued*, so the "time" is roughly constant and absurdly small. Only the
    synchronised numbers grow with n^3 the way the arithmetic says they must.
    """
    backend = make_backend("torch-cuda")
    rows = []
    for n in sizes:
        import numpy as np

        rng = np.random.default_rng(0)
        a, b = (backend.from_numpy(x, dtype) for x in MATMUL.make_inputs(rng, n, dtype))
        fn = lambda: a @ b  # noqa: E731
        right = summarize(measure(fn, backend.sync, repeats=20))
        # The wrong way: no sync inside the loop (one at the very end only).
        wrong = summarize(measure(fn, lambda: None, repeats=20))
        backend.sync()
        rows.append(
            {
                "n": n,
                "with_sync_ms": right.median * 1e3,
                "without_sync_ms": wrong.median * 1e3,
                "apparent_speedup": right.median / wrong.median,
                "apparent_tflops_without_sync": MATMUL.flops(n) / wrong.median / 1e12,
            }
        )
    return pd.DataFrame(rows)
