"""Command line interface.

    python -m gpubench env                         # what hardware is visible
    python -m gpubench run --preset full --out results/colab-t4
    python -m gpubench report results/colab-t4     # figures + summary.md
    python -m gpubench cheats                      # show the anti-cheat guards working
"""

from __future__ import annotations

import argparse
import json
import sys

from .backends import available_backends


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_env(_args) -> int:
    from .runner import environment

    print(json.dumps(environment(), indent=2))
    print("available backends:", ", ".join(available_backends()))
    return 0


def cmd_run(args) -> int:
    from .runner import SuiteConfig, run_suite

    cfg = SuiteConfig(preset=args.preset, repeats=args.repeats, seed=args.seed,
                      skip_transfer=args.skip_transfer, skip_memory=args.skip_memory)
    if args.workloads:
        cfg.workloads = _split(args.workloads)
    if args.backends:
        cfg.backends = _split(args.backends)
    if args.dtypes:
        cfg.dtypes = _split(args.dtypes)
    run_suite(cfg, args.out)
    if args.report:
        from .report import build_report

        build_report(args.out)
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    path = build_report(args.run_dir)
    print(f"Report written to {path}")
    return 0


def cmd_cheats(args) -> int:
    from .demo import cheat_demo, cheat_demo_passed

    df = cheat_demo(backend_name=args.backend, size=args.size)
    cols = ["backend", "implementation", "time_median", "gflops", "status", "reasons"]
    print(df[cols].to_string(index=False))
    passed = cheat_demo_passed(df)
    print("\nAll cheats caught, honest run accepted." if passed else "\nGUARD FAILURE")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpubench", description="Honest CPU vs GPU benchmarking.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print hardware / library info").set_defaults(func=cmd_env)

    r = sub.add_parser("run", help="run the benchmark suite")
    r.add_argument("--preset", choices=["quick", "full"], default="quick")
    r.add_argument("--out", default="results/latest")
    r.add_argument("--workloads", help="comma separated, e.g. matmul,axpy")
    r.add_argument("--backends", help="comma separated: numpy,torch-cpu,torch-cuda")
    r.add_argument("--dtypes", help="comma separated: float32,float16")
    r.add_argument("--repeats", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--skip-transfer", action="store_true")
    r.add_argument("--skip-memory", action="store_true")
    r.add_argument("--report", action="store_true", help="build the report afterwards")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="build figures and summary.md from a run")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    c = sub.add_parser("cheats", help="demonstrate that broken implementations are caught")
    c.add_argument("--backend", default=None, help="default: fastest available")
    c.add_argument("--size", type=int, default=1024)
    c.set_defaults(func=cmd_cheats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
