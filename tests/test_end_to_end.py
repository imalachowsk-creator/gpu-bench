"""Run the whole pipeline on the CPU: suite -> CSV -> report."""

import json

import pandas as pd

from gpubench.cli import main
from gpubench.demo import cheat_demo, cheat_demo_passed
from gpubench.runner import PRESETS, SuiteConfig, run_suite


def test_quick_suite_and_report(tmp_path):
    cfg = SuiteConfig(preset="quick", backends=["numpy"], dtypes=["float32"],
                      repeats=5, max_seconds=0.2)
    out = run_suite(cfg, tmp_path / "run", log=lambda _: None)

    df = pd.read_csv(out / "compute.csv")
    expected_rows = sum(len(PRESETS["quick"][w]) for w in cfg.workloads)
    assert len(df) == expected_rows
    assert (df["status"] == "ok").all(), df.loc[df["status"] != "ok", "reasons"].tolist()
    assert (df["time_median"] > 0).all()
    assert json.loads((out / "env.json").read_text())["numpy"]

    assert main(["report", str(out)]) == 0
    summary = (out / "summary.md").read_text()
    assert "Benchmark summary" in summary
    assert (out / "figures" / "time_vs_size.png").stat().st_size > 10_000


def test_cheat_demo_catches_everything():
    df = cheat_demo(backend_name="numpy", size=256)
    assert cheat_demo_passed(df), df[["implementation", "status", "reasons"]]


def test_cli_env_runs(capsys):
    assert main(["env"]) == 0
    assert "available backends" in capsys.readouterr().out
