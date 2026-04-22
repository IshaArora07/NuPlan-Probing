#!/usr/bin/env python3
"""
analyse_improvement.py — Scripts 11–12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# FIX: correct __file__
sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_all_metrics,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, PALETTE,
    PROGRESS_FAIL_THRESH,
)

set_plot_style()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _load_merged_with_type(run_name: str) -> pd.DataFrame:
    merged = load_all_metrics(run_name)

    if get_scenario_type_col(merged) is None:
        try:
            rr = load_runner_report(run_name)
            token_col = merged.columns[0]

            if token_col in rr.columns and "scenario_type" in rr.columns:
                merged = pd.merge(
                    merged,
                    rr[[token_col, "scenario_type"]],
                    on=token_col,
                    how="left",
                )
        except Exception:
            pass

    return merged


def _failure_rate(series: pd.Series, threshold: float = 0.5):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return (s < threshold).mean()


def _severity(rate):
    if np.isnan(rate):
        return "unknown"
    if rate >= 0.4:
        return "CRITICAL"
    if rate >= 0.2:
        return "HIGH"
    if rate >= 0.1:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# SCRIPT 11 — Braking
# ─────────────────────────────────────────────

def script_11_braking_analysis(run_name: str):
    print(f"\n=== SCRIPT 11 — Braking [{run_name}] ===")

    merged = _load_merged_with_type(run_name)
    total = len(merged)

    if "ego_lon_acceleration" not in merged.columns:
        print("[WARN] Missing ego_lon_acceleration")
        return merged

    # classify
    merged["harsh_brake"] = merged["ego_lon_acceleration"] < 0.5

    n = merged["harsh_brake"].sum()
    print(f"Harsh braking: {n}/{total} ({100*n/total:.1f}%)")

    # TTC relation
    if "time_to_collision_within_bound" in merged.columns:
        phantom = (merged["harsh_brake"]) & (merged["time_to_collision_within_bound"] > 0.8)
        reactive = (merged["harsh_brake"]) & (merged["time_to_collision_within_bound"] < 0.5)

        print(f"Phantom braking: {phantom.sum()} ({100*phantom.mean():.1f}%)")
        print(f"Reactive braking: {reactive.sum()} ({100*reactive.mean():.1f}%)")

    return merged


# ─────────────────────────────────────────────
# SCRIPT 12 — Recommendations
# ─────────────────────────────────────────────

KEY_METRICS = [
    "ego_is_comfortable",
    "ego_is_making_progress",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
]

def script_12_t8_recommendations(run_name: str, compare_runs=None):
    print(f"\n=== SCRIPT 12 — T8 Recommendations [{run_name}] ===")

    merged = _load_merged_with_type(run_name)

    token_col = merged.columns[0]
    sc_col = get_scenario_type_col(merged)

    # global means
    metrics = [m for m in KEY_METRICS if m in merged.columns]
    means = {m: merged[m].mean() for m in metrics}

    print("\nGlobal metric means:")
    for k, v in means.items():
        print(f"{k:<40} {v:.3f}")

    # failure ranking
    ranking = []
    for m in metrics:
        fr = _failure_rate(merged[m])
        ranking.append((m, fr, _severity(fr)))

    ranking.sort(key=lambda x: x[1], reverse=True)

    print("\nFailure ranking:")
    for m, fr, sev in ranking:
        print(f"{m:<40} {fr:.3f}  {sev}")

    # simple recommendations
    print("\nTop recommendations:")

    actions = []

    if means.get("ego_is_making_progress", 1) < 0.75:
        actions.append("Increase forward anchors / reduce conservative bias")

    if means.get("ego_is_comfortable", 1) < 0.8:
        actions.append("Increase smoothness / yaw penalties")

    if means.get("no_ego_at_fault_collisions", 1) < 0.9:
        actions.append("Improve turn anchors + collision loss weight")

    if means.get("time_to_collision_within_bound", 1) < 0.85:
        actions.append("Strengthen TTC / ESDF safety modeling")

    for i, a in enumerate(actions[:3], 1):
        print(f"{i}. {a}")

    # save CSV
    outdir = get_output_dir() / "improvement"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(ranking, columns=["metric", "fail_rate", "severity"])
    df.to_csv(outdir / f"s12_priority_{run_name}.csv", index=False)

    print(f"\nSaved → {outdir}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all(run_name, compare_runs=None):
    script_11_braking_analysis(run_name)
    script_12_t8_recommendations(run_name, compare_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", default=PRIMARY_RUN)
    parser.add_argument("--script", type=int, default=0)
    parser.add_argument("--runs", nargs="+", default=None)

    args = parser.parse_args()

    if args.script == 0:
        run_all(args.run, args.runs)
    elif args.script == 11:
        script_11_braking_analysis(args.run)
    elif args.script == 12:
        script_12_t8_recommendations(args.run, args.runs)
    else:
        print("Invalid script")
