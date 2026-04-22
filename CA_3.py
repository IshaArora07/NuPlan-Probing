"""
analyse_improvement.py — Scripts 11–12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, PALETTE,
    METRIC_FILES,   # FIXED: missing import
    PROGRESS_FAIL_THRESH
)

set_plot_style()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _load_merged_with_type(run_name: str, metrics=None):
    rr = load_runner_report(run_name)

    token_col = rr.columns[0]
    sc_col = get_scenario_type_col(rr)

    base = rr[[token_col] + ([sc_col] if sc_col else [])].copy()
    del rr

    for metric_name in (metrics or METRIC_FILES):
        df = load_metric(run_name, metric_name)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        sub = df[[tok, num_cols[0]]].rename(columns={num_cols[0]: metric_name})
        base = base.merge(sub, left_on=token_col, right_on=tok, how="left")

        if tok != token_col and tok in base.columns:
            base = base.drop(columns=[tok])

        del df, sub

    return base


def _failure_rate(series):
    s = series.dropna()
    return np.nan if len(s) == 0 else (s < 0.5).mean()


# ─────────────────────────────────────────────
# SCRIPT 11
# ─────────────────────────────────────────────

def script_11_braking_analysis(run_name: str):
    print(f"\nSCRIPT 11 — Braking [{run_name}]")

    metrics = [
        "ego_lon_acceleration",
        "ego_lon_jerk",
        "time_to_collision_within_bound",
        "ego_is_comfortable"
    ]

    df = _load_merged_with_type(run_name, metrics)

    if "ego_lon_acceleration" not in df:
        print("Missing lon acceleration")
        return df

    df["harsh"] = df["ego_lon_acceleration"] < 0.5

    print("Harsh braking rate:", df["harsh"].mean())

    if "time_to_collision_within_bound" in df:
        df["phantom"] = df["harsh"] & (df["time_to_collision_within_bound"] > 0.8)
        print("Phantom braking:", df["phantom"].mean())

    return df


# ─────────────────────────────────────────────
# SCRIPT 12
# ─────────────────────────────────────────────

def script_12_t8_recommendations(run_name: str, runs_for_comparison=None):
    print(f"\nSCRIPT 12 — Recommendations [{run_name}]")

    # IMPORTANT: restrict metrics → avoid OOM
    key_metrics = [
        "ego_is_comfortable",
        "ego_is_making_progress",
        "no_ego_at_fault_collisions",
        "time_to_collision_within_bound",
        "ego_lon_acceleration",
        "ego_lon_jerk",
    ]

    df = _load_merged_with_type(run_name, key_metrics)

    results = []

    for m in key_metrics:
        if m not in df:
            continue

        fr = _failure_rate(df[m])

        results.append({
            "metric": m,
            "mean": df[m].mean(),
            "fail_rate": fr
        })

    summary = pd.DataFrame(results).sort_values("fail_rate", ascending=False)

    print("\nPriority metrics:")
    print(summary)

    # Top 3 recommendations
    print("\nTop issues:")
    for _, row in summary.head(3).iterrows():
        print(f" - {row['metric']} (fail rate {row['fail_rate']:.2f})")

    out = get_output_dir() / f"s12_summary_{run_name}.csv"
    summary.to_csv(out, index=False)

    print("Saved:", out)

    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all_improvement(run_name: str, compare_runs=None):
    script_11_braking_analysis(run_name)
    script_12_t8_recommendations(run_name, compare_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=PRIMARY_RUN)
    parser.add_argument("--script", type=int, default=0)
    parser.add_argument("--runs", nargs="+", default=None)

    args = parser.parse_args()

    if args.script == 0:
        run_all_improvement(args.run, args.runs)
    elif args.script == 11:
        script_11_braking_analysis(args.run)
    elif args.script == 12:
        script_12_t8_recommendations(args.run, args.runs)
    else:
        print("Invalid script number")
