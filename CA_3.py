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

# ── imports ─────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

import importlib

for _name in ["load_utils", "00_load_utils"]:
    try:
        module = importlib.import_module(_name)
        sys.modules["load_utils"] = module
        break
    except ModuleNotFoundError:
        continue

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, PALETTE,
    METRIC_FILES,
    PROGRESS_FAIL_THRESH,
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

    metrics = metrics or METRIC_FILES

    for m in metrics:
        df = load_metric(run_name, m)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        sub = df[[tok, num_cols[0]]].rename(columns={num_cols[0]: m})
        base = base.merge(sub, left_on=token_col, right_on=tok, how="left")

        if tok != token_col and tok in base.columns:
            base = base.drop(columns=[tok])

        del df, sub

    return base


def _failure_rate(series, threshold=0.5):
    s = series.dropna()
    return (s < threshold).mean() if len(s) else np.nan


def _severity(rate):
    if np.isnan(rate): return "unknown"
    if rate >= 0.4: return "CRITICAL"
    if rate >= 0.2: return "HIGH"
    if rate >= 0.1: return "MEDIUM"
    return "LOW"


def _severity_color(rate):
    if np.isnan(rate): return "#555"
    if rate >= 0.4: return "#FF6B6B"
    if rate >= 0.2: return "#F28C38"
    if rate >= 0.1: return "#F5C542"
    return "#6BCB77"


# ─────────────────────────────────────────────
# SCRIPT 11 — Braking
# ─────────────────────────────────────────────

def script_11_braking_analysis(run_name: str):
    print("\nSCRIPT 11 — Braking Analysis")

    merged = _load_merged_with_type(run_name, [
        "ego_lon_acceleration",
        "ego_lon_jerk",
        "time_to_collision_within_bound",
        "ego_is_comfortable",
        "ego_is_making_progress",
    ])

    total = len(merged)

    if "ego_lon_acceleration" not in merged:
        print("No lon_acc data")
        return merged

    merged["braking_harsh"] = merged["ego_lon_acceleration"] < 0.5

    harsh = merged["braking_harsh"].mean()
    print(f"Harsh braking rate: {harsh:.3f}")

    if "time_to_collision_within_bound" in merged:
        phantom = merged["braking_harsh"] & (merged["time_to_collision_within_bound"] >= 0.8)
        reactive = merged["braking_harsh"] & (merged["time_to_collision_within_bound"] < 0.5)

        print(f"Phantom: {phantom.mean():.3f}")
        print(f"Reactive: {reactive.mean():.3f}")

    return merged


# ─────────────────────────────────────────────
# SCRIPT 12 — Recommendations
# ─────────────────────────────────────────────

METRICS = [
    "ego_is_comfortable",
    "ego_is_making_progress",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
]

def script_12_t8_recommendations(run_name: str, runs_for_comparison=None):
    print("\nSCRIPT 12 — T8 Recommendations")

    merged = _load_merged_with_type(run_name)

    token_col = merged.columns[0]
    sc_col = get_scenario_type_col(merged)

    global_means = {m: merged[m].mean() for m in merged.columns if m != token_col}

    # ── failure ranking
    rows = []
    for m in global_means:
        fr = _failure_rate(merged[m]) if m in merged else np.nan
        rows.append((m, fr, _severity(fr)))

    df = pd.DataFrame(rows, columns=["metric", "fail_rate", "severity"])
    df = df.sort_values("fail_rate", ascending=False)

    print("\nTop failing metrics:")
    print(df.head(10))

    # ── scenario type breakdown
    if sc_col:
        type_means = merged.groupby(sc_col).mean(numeric_only=True)
        print("\nWorst scenario types:")
        print(type_means.mean(axis=1).sort_values().head())

    # ── plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["metric"], df["fail_rate"],
           color=[_severity_color(x) for x in df["fail_rate"]])
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, f"s12_{run_name}", subdir="improvement")

    return df


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
        print("Invalid script")
