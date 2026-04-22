#!/usr/bin/env python3
"""
run_all.py — EMoE Full Analysis Pipeline
"""

import argparse
import sys
from pathlib import Path

# FIX: correct __file__
sys.path.insert(0, str(Path(__file__).parent))

import analyse_core
import analyse_behaviour
import analyse_improvement

from load_utils import PRIMARY_RUN, RUN_PATHS, get_output_dir, set_plot_style

set_plot_style()


def main():
    parser = argparse.ArgumentParser(description="EMoE Full Analysis Pipeline")

    parser.add_argument(
        "--run",
        default=PRIMARY_RUN,
        help="Primary run to analyse"
    )

    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Runs for cross-run comparison (default: all)"
    )

    parser.add_argument(
        "--group",
        choices=["core", "behaviour", "improvement", "all"],
        default="all",
        help="Which group to run"
    )

    args = parser.parse_args()

    run = args.run
    compare_runs = args.runs if args.runs is not None else list(RUN_PATHS.keys())

    print("\n" + "=" * 65)
    print("  EMoE Analysis Pipeline")
    print(f"  Primary run  : {run}")
    print(f"  Compare runs : {compare_runs}")
    print(f"  Group        : {args.group}")
    print(f"  Output dir   : {get_output_dir()}")
    print("=" * 65 + "\n")

    # ─────────────────────────────
    # CORE
    # ─────────────────────────────
    if args.group in ("core", "all"):
        try:
            analyse_core.run_all_core(run)
        except Exception as e:
            print(f"[ERROR] Core failed: {e}")

    # ─────────────────────────────
    # BEHAVIOUR
    # ─────────────────────────────
    if args.group in ("behaviour", "all"):
        try:
            analyse_behaviour.run_all_behaviour(run, compare_runs)
        except Exception as e:
            print(f"[ERROR] Behaviour failed: {e}")

    # ─────────────────────────────
    # IMPROVEMENT
    # ─────────────────────────────
    if args.group in ("improvement", "all"):
        try:
            analyse_improvement.run_all_improvement(run, compare_runs)
        except Exception as e:
            print(f"[ERROR] Improvement failed: {e}")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Outputs in: {get_output_dir()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
