#!/usr/bin/env python3
"""
analyse_behaviour.py — Scripts 6–10
EMoE Simulation Behavioural Deep-Dive
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# FIX: correct __file__ usage
sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_all_metrics,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, PALETTE,
    HARD_BRAKE_THRESHOLD, HIGH_LAT_ACCEL, HIGH_YAW_RATE,
    PROGRESS_FAIL_THRESH, TTC_FAIL_THRESH,
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


def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────
# SCRIPT 6 — Comfort correlations
# ─────────────────────────────────────────────

def script_6_comfort_analysis(run_name: str):
    print(f"\n=== SCRIPT 6 — Comfort [{run_name}] ===")

    merged = _load_merged_with_type(run_name)

    if "ego_is_comfortable" not in merged.columns:
        print("[WARN] Missing ego_is_comfortable")
        return

    metrics = [m for m in [
        "ego_jerk",
        "ego_lat_acceleration",
        "ego_lon_acceleration",
        "ego_yaw_rate",
    ] if m in merged.columns]

    rows = []
    for m in metrics:
        sub = merged[["ego_is_comfortable", m]].dropna()
        if len(sub) < 10:
            continue

        corr = sub["ego_is_comfortable"].corr(sub[m])
        rows.append({
            "metric": m,
            "corr": round(corr, 4)
        })

    df = pd.DataFrame(rows).sort_values("corr", ascending=False)
    print(df.to_string(index=False))

    return merged, df


# ─────────────────────────────────────────────
# SCRIPT 7 — Progress vs Safety
# ─────────────────────────────────────────────

def script_7_progress_vs_safety(runs=None):
    print("\n=== SCRIPT 7 — Progress vs Safety ===")

    if runs is None:
        runs = list(RUN_PATHS.keys())

    results = []

    for run in runs:
        try:
            merged = _load_merged_with_type(run)

            s_cols = [c for c in [
                "no_ego_at_fault_collisions",
                "time_to_collision_within_bound"
            ] if c in merged.columns]

            p_cols = [c for c in [
                "ego_is_making_progress",
                "ego_progress_along_expert_route"
            ] if c in merged.columns]

            if not s_cols or not p_cols:
                continue

            merged["safety"] = merged[s_cols].mean(axis=1)
            merged["progress"] = merged[p_cols].mean(axis=1)

            results.append({
                "run": run,
                "safety_mean": merged["safety"].mean(),
                "progress_mean": merged["progress"].mean()
            })

        except Exception as e:
            print(f"[WARN] {run}: {e}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    return df


# ─────────────────────────────────────────────
# SCRIPT 8 — Stopping behaviour
# ─────────────────────────────────────────────

def script_8_stopping_analysis(run_name: str):
    print(f"\n=== SCRIPT 8 — Stopping [{run_name}] ===")

    merged = _load_merged_with_type(run_name)

    if "ego_is_making_progress" not in merged.columns:
        print("[WARN] Missing progress metric")
        return

    stopped = merged["ego_is_making_progress"] < PROGRESS_FAIL_THRESH

    total = len(merged)
    n_stop = stopped.sum()

    print(f"Stopped: {n_stop} / {total} ({100*n_stop/total:.1f}%)")

    return merged


# ─────────────────────────────────────────────
# SCRIPT 9 — Runner health
# ─────────────────────────────────────────────

def script_9_runner_errors(run_name: str):
    print(f"\n=== SCRIPT 9 — Infra [{run_name}] ===")

    rr = load_runner_report(run_name)

    total = len(rr)
    print(f"Total: {total}")

    status_col = next((c for c in rr.columns if "status" in c.lower()), None)

    if status_col:
        counts = rr[status_col].value_counts()
        print(counts)

    return rr


# ─────────────────────────────────────────────
# SCRIPT 10 — HTML dashboard
# ─────────────────────────────────────────────

def script_10_html_dashboard(run_name: str):
    print(f"\n=== SCRIPT 10 — HTML [{run_name}] ===")

    out = get_output_dir()
    path = out / f"behaviour_dashboard_{run_name}.html"

    html = f"""
    <html>
    <body style="background:#111;color:white;font-family:monospace">
    <h1>EMoE Behaviour — {run_name}</h1>
    <p>Generated dashboard</p>
    </body>
    </html>
    """

    path.write_text(html)
    print(f"Saved → {path}")

    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all(run_name, compare_runs=None):
    script_6_comfort_analysis(run_name)
    script_7_progress_vs_safety(compare_runs)
    script_8_stopping_analysis(run_name)
    script_9_runner_errors(run_name)
    script_10_html_dashboard(run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", default=PRIMARY_RUN)
    parser.add_argument("--script", type=int, default=0)
    parser.add_argument("--runs", nargs="+", default=None)

    args = parser.parse_args()

    if args.script == 0:
        run_all(args.run, args.runs)
    elif args.script == 6:
        script_6_comfort_analysis(args.run)
    elif args.script == 7:
        script_7_progress_vs_safety(args.runs)
    elif args.script == 8:
        script_8_stopping_analysis(args.run)
    elif args.script == 9:
        script_9_runner_errors(args.run)
    elif args.script == 10:
        script_10_html_dashboard(args.run)
    else:
        print("Invalid script")
