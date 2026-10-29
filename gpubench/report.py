"""Turn raw CSV results into figures and a Markdown summary.

    python -m gpubench report results/<run>

Creates ``figures/*.png`` and ``summary.md`` inside the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Colab, servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- style -----------------------------------------------------------------
# Colour follows the *series identity*, never its position, so the same
# backend/dtype has the same colour in every figure.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES_COLORS = {
    "numpy float32": "#2a78d6",
    "torch-cpu float32": "#eb6834",
    "torch-cuda float32": "#1baf7a",
    "torch-cuda float16": "#eda100",
}
EXTRA_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
WORKLOAD_TITLES = {
    "matmul": "Matrix multiply (compute-bound)",
    "axpy": "axpy  y = a*x + y  (memory-bound)",
    "mean": "Mean (reduction)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
        }
    )


def _color(label: str, cache: dict[str, str]) -> str:
    if label in SERIES_COLORS:
        return SERIES_COLORS[label]
    if label not in cache:
        cache[label] = EXTRA_COLORS[len(cache) % len(EXTRA_COLORS)]
    return cache[label]


def _series_label(row) -> str:
    return f"{row['backend']} {row['dtype']}"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- figures -----------------------------------------------------------------

def plot_times(df: pd.DataFrame, out: Path) -> Path:
    """Median time vs size, log-log, with the IQR as a band."""
    workloads = [w for w in WORKLOAD_TITLES if w in set(df["workload"])]
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.6 * len(workloads), 3.8), squeeze=False)
    extra: dict[str, str] = {}
    for ax, wname in zip(axes[0], workloads):
        sub = df[(df["workload"] == wname) & (df["status"] == "ok")].copy()
        sub["series"] = sub.apply(_series_label, axis=1)
        for label, g in sub.groupby("series", sort=False):
            g = g.sort_values("size")
            c = _color(label, extra)
            ax.fill_between(g["size"], g["time_q1"] * 1e3, g["time_q3"] * 1e3, color=c, alpha=0.18, lw=0)
            ax.plot(g["size"], g["time_median"] * 1e3, marker="o", color=c, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(WORKLOAD_TITLES[wname], loc="left")
        ax.set_xlabel(_size_label(wname))
        ax.set_ylabel("median time [ms] (band = IQR)")
    axes[0][0].legend(loc="upper left")
    return _save(fig, out / "time_vs_size.png")


def _size_label(wname: str) -> str:
    return "matrix size n (n x n)" if wname == "matmul" else "vector length n"


def speedup_table(df: pd.DataFrame, baseline: str = "numpy") -> pd.DataFrame:
    """Speed-up of every series relative to the float32 baseline backend."""
    ok = df[df["status"] == "ok"]
    base = ok[(ok["backend"] == baseline) & (ok["dtype"] == "float32")][["workload", "size", "time_median"]]
    base = base.rename(columns={"time_median": "baseline_time"})
    other = ok[ok["backend"] != baseline]
    merged = other.merge(base, on=["workload", "size"], how="inner")
    merged["speedup"] = merged["baseline_time"] / merged["time_median"]
    merged["series"] = merged.apply(_series_label, axis=1)
    return merged


def plot_speedup(df: pd.DataFrame, out: Path, baseline: str = "numpy") -> Path | None:
    sp = speedup_table(df, baseline)
    if sp.empty:
        return None
    workloads = [w for w in WORKLOAD_TITLES if w in set(sp["workload"])]
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.6 * len(workloads), 3.8), squeeze=False)
    extra: dict[str, str] = {}
    for ax, wname in zip(axes[0], workloads):
        sub = sp[sp["workload"] == wname]
        ax.axhline(1.0, color=INK_2, lw=1, ls="--")
        ax.text(sub["size"].min(), 1.0, f" {baseline} float32 = 1x", color=INK_2, fontsize=7, va="bottom")
        for label, g in sub.groupby("series", sort=False):
            g = g.sort_values("size")
            ax.plot(g["size"], g["speedup"], marker="o", color=_color(label, extra), label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(WORKLOAD_TITLES[wname], loc="left")
        ax.set_xlabel(_size_label(wname))
        ax.set_ylabel(f"speed-up vs {baseline} (higher is better)")
    axes[0][0].legend(loc="upper left")
    return _save(fig, out / "speedup.png")


def plot_throughput(df: pd.DataFrame, out: Path) -> Path:
    """Matmul in GFLOP/s (compute) next to axpy in GB/s (memory bandwidth)."""
    panels = [("matmul", "gflops", "GFLOP/s"), ("axpy", "gb_per_s", "GB/s")]
    panels = [p for p in panels if p[0] in set(df["workload"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.8), squeeze=False)
    extra: dict[str, str] = {}
    for ax, (wname, col, unit) in zip(axes[0], panels):
        sub = df[(df["workload"] == wname) & (df["status"] == "ok")].copy()
        sub["series"] = sub.apply(_series_label, axis=1)
        for label, g in sub.groupby("series", sort=False):
            g = g.sort_values("size")
            ax.plot(g["size"], g[col], marker="o", color=_color(label, extra), label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{wname}: achieved {unit}", loc="left")
        ax.set_xlabel(_size_label(wname))
        ax.set_ylabel(unit)
    axes[0][0].legend(loc="upper left")
    return _save(fig, out / "throughput.png")


def plot_transfer(tr: pd.DataFrame, fits: pd.DataFrame, out: Path) -> Path:
    colors = {"pinned": "#2a78d6", "pageable": "#eb6834"}
    titles = {"H2D": "Host -> GPU", "D2H": "GPU -> host"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, direction in zip(axes, ("H2D", "D2H")):
        for memory, g in tr[tr["direction"] == direction].groupby("memory"):
            g = g.sort_values("bytes")
            c = colors.get(memory, INK_2)
            ax.plot(g["bytes"], g["gb_per_s"], marker="o", color=c, label=f"{memory} memory")
            fit = fits[(fits["direction"] == direction) & (fits["memory"] == memory)]
            if not fit.empty:
                lat = fit["latency_us"].iloc[0] * 1e-6
                bw = fit["bandwidth_gb_per_s"].iloc[0] * 1e9
                x = np.geomspace(g["bytes"].min(), g["bytes"].max(), 100)
                ax.plot(x, x / (lat + x / bw) / 1e9, color=c, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{titles[direction]} copy (dashed = fitted model)", loc="left")
        ax.set_xlabel("transfer size [bytes]")
    axes[0].set_ylabel("effective GB/s")
    axes[0].legend(loc="upper left")
    return _save(fig, out / "transfer.png")


def plot_memory(mem: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colors = {"float32": "#2a78d6", "float16": "#eb6834"}
    for dtype, g in mem.groupby("dtype"):
        ax.scatter(g["measured_mib"], g["predicted_mib"], s=40, color=colors.get(dtype, INK_2),
                   edgecolor=SURFACE, linewidth=1.5, label=dtype, zorder=3)
    lim = [mem[["measured_mib", "predicted_mib"]].min().min() * 0.8,
           mem[["measured_mib", "predicted_mib"]].max().max() * 1.25]
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--")
    ax.text(lim[0] * 1.2, lim[0] * 1.6, "perfect prediction", color=INK_2, fontsize=7,
            ha="left", va="bottom", rotation=45, rotation_mode="anchor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Peak GPU memory: predicted vs measured", loc="left")
    ax.set_xlabel("measured [MiB]")
    ax.set_ylabel("predicted [MiB]")
    ax.legend(loc="upper left")
    return _save(fig, out / "memory.png")


# --- summary -----------------------------------------------------------------

def crossover_sizes(df: pd.DataFrame, gpu_backend: str = "torch-cuda", baseline: str = "numpy") -> pd.DataFrame:
    """Smallest size from which the GPU (float32) beats the CPU baseline for good."""
    sp = speedup_table(df, baseline)
    sp = sp[(sp["backend"] == gpu_backend) & (sp["dtype"] == "float32")]
    rows = []
    for wname, g in sp.groupby("workload"):
        g = g.sort_values("size")
        faster = g["speedup"].to_numpy() > 1.0
        cross = None
        for i in range(len(faster)):
            if faster[i:].all():
                cross = int(g["size"].iloc[i])
                break
        rows.append({
            "workload": wname,
            "gpu_wins_from_size": cross if cross is not None else "never (in tested range)",
            "max_speedup": round(float(g["speedup"].max()), 1),
            "at_size": int(g.loc[g["speedup"].idxmax(), "size"]),
        })
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def build_report(run_dir: str | Path) -> Path:
    _style()
    run = Path(run_dir)
    figs = run / "figures"
    figs.mkdir(exist_ok=True)

    env = json.loads((run / "env.json").read_text(encoding="utf-8")) if (run / "env.json").exists() else {}
    df = pd.read_csv(run / "compute.csv")

    lines = ["# Benchmark summary", ""]
    hw = env.get("gpu") or "no GPU"
    lines += [
        f"- **GPU:** {hw}" + (f" ({env.get('gpu_memory_gib')} GiB, CUDA {env.get('cuda')})" if env.get("gpu") else ""),
        f"- **CPU:** {env.get('processor', '?')}  |  **Platform:** {env.get('platform', '?')}",
        f"- **Libraries:** numpy {env.get('numpy')}, torch {env.get('torch')}",
        f"- **Run at:** {env.get('timestamp_utc', '?')}",
        "",
    ]

    plot_times(df, figs)
    lines += ["## Time vs problem size", "", "![time vs size](figures/time_vs_size.png)", ""]

    if plot_speedup(df, figs):
        lines += ["## Speed-up vs NumPy (float32)", "", "![speed-up](figures/speedup.png)", ""]
        cross = crossover_sizes(df)
        if not cross.empty:
            lines += ["### Where the GPU starts to win (float32)", "", _md(cross), ""]

    plot_throughput(df, figs)
    lines += ["## Achieved throughput", "", "![throughput](figures/throughput.png)", ""]
    peaks = (
        df[df["status"] == "ok"]
        .groupby(["workload", "backend", "dtype"])
        .agg(peak_gflops=("gflops", "max"), peak_gb_per_s=("gb_per_s", "max"))
        .reset_index()
    )
    lines += [_md(peaks), ""]

    if (run / "transfer.csv").exists():
        from .transfer import fit_table

        tr = pd.read_csv(run / "transfer.csv")
        fits = pd.DataFrame(fit_table(tr.to_dict("records")))
        plot_transfer(tr, fits, figs)
        lines += [
            "## Host <-> device transfers",
            "",
            "Model: `time = latency + bytes / bandwidth`, fitted by weighted least squares.",
            "",
            "![transfer](figures/transfer.png)",
            "",
            _md(fits),
            "",
        ]

    if (run / "memory.csv").exists():
        mem = pd.read_csv(run / "memory.csv")
        plot_memory(mem, figs)
        lines += ["## Peak memory: prediction vs measurement", "", "![memory](figures/memory.png)", "", _md(mem), ""]

    lines += ["## Verification", ""]
    counts = df["status"].value_counts().rename_axis("status").reset_index(name="cases")
    lines += [_md(counts), ""]
    bad = df[df["status"] != "ok"]
    if bad.empty:
        lines += ["Every timed result passed correctness, fresh-input and plausibility checks.", ""]
    else:
        cols = ["workload", "backend", "dtype", "size", "status", "reasons"]
        lines += ["Flagged cases (excluded from the figures):", "", _md(bad[cols]), ""]

    path = run / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
