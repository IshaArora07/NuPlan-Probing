#!/usr/bin/env python3
"""
run_all.py — EMoE Full Analysis Pipeline
Runs all scripts: core → behaviour → improvement
"""

import argparse
import sys
from pathlib import Path

# FIX: correct __file__
sys.path.insert(0, str(Path(__file__).parent))

from load_utils import PRIMARY_RUN, RUN_PATHS, get_output_dir, set_plot_style

set_plot_style()


def main():
    parser = argparse.ArgumentParser(description="EMoE Full Analysis Pipeline")

    parser.add_argument(
        "--run",
        default=PRIMARY_RUN,
        help="Primary run to analyse (default: PRIMARY_RUN in load_utils.py)"
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Runs to include in cross-run comparisons (default: all in RUN_PATHS)"
    )

    parser.add_argument(
        "--group",
        choices=["core", "behaviour", "improvement", "all"],
        default="all",
        help="Which group of scripts to run"
    )

    args = parser.parse_args()

    run = args.run
    compare_runs = args.runs or list(RUN_PATHS.keys())

    # Optional safety check
    if run not in RUN_PATHS:
        print(f"[WARN] '{run}' not in RUN_PATHS. Continuing anyway.")

    print("\n" + "=" * 65)
    print("  EMoE Analysis Pipeline")
    print(f"  Primary run  : {run}")
    print(f"  Compare runs : {compare_runs}")
    print(f"  Group        : {args.group}")
    print(f"  Output dir   : {get_output_dir()}")
    print("=" * 65 + "\n")

    # ─────────────────────────────────────
    # CORE (1–5)
    # ─────────────────────────────────────
    if args.group in ("core", "all"):
        try:
            from analyse_core import run_all_core
            run_all_core(run, compare_runs=compare_runs)
        except Exception as e:
            print(f"[ERROR] Core pipeline failed: {e}")

    # ─────────────────────────────────────
    # BEHAVIOUR (6–10)
    # ─────────────────────────────────────
    if args.group in ("behaviour", "all"):
        try:
            from analyse_behaviour import run_all_behaviour
            run_all_behaviour(run, compare_runs=compare_runs)
        except Exception as e:
            print(f"[ERROR] Behaviour pipeline failed: {e}")

    # ─────────────────────────────────────
    # IMPROVEMENT (11–12)
    # ─────────────────────────────────────
    if args.group in ("improvement", "all"):
        try:
            from analyse_improvement import run_all_improvement
            run_all_improvement(run, compare_runs=compare_runs)
        except Exception as e:
            print(f"[ERROR] Improvement pipeline failed: {e}")

    print("\n" + "=" * 65)
    print("  Pipeline complete. Outputs in:")
    print(f"  {get_output_dir()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
