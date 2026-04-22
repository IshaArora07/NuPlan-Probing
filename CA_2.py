"""
analyse_behaviour.py — Scripts 6–10
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, bar_chart, PALETTE,
    METRIC_FILES,
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


def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────
# SCRIPT 6
# ─────────────────────────────────────────────

def script_6_comfort_analysis(run_name: str):
    print(f"\nSCRIPT 6 — Comfort [{run_name}]")

    metrics = ["ego_jerk", "ego_lat_acceleration", "ego_lon_acceleration",
               "ego_yaw_rate"]

    merged = _load_merged_with_type(run_name, metrics + ["ego_is_comfortable"])

    if "ego_is_comfortable" not in merged.columns:
        print("Missing ego_is_comfortable")
        return

    rows = []
    for m in metrics:
        if m not in merged.columns:
            continue

        sub = merged[[m, "ego_is_comfortable"]].dropna()
        if len(sub) < 10:
            continue

        rows.append({
            "metric": m,
            "corr": sub[m].corr(sub["ego_is_comfortable"])
        })

    df = pd.DataFrame(rows).sort_values("corr", ascending=False)
    print(df)

    return merged, df


# ─────────────────────────────────────────────
# SCRIPT 7
# ─────────────────────────────────────────────

def script_7_progress_vs_safety(runs=None):
    print("\nSCRIPT 7 — Progress vs Safety")

    runs = runs or list(RUN_PATHS.keys())
    results = {}

    for run in runs:
        try:
            df = _load_merged_with_type(run, [
                "no_ego_at_fault_collisions",
                "ego_is_making_progress"
            ])

            df["safety"] = df["no_ego_at_fault_collisions"]
            df["progress"] = df["ego_is_making_progress"]

            results[run] = df
        except:
            continue

    print(f"Loaded {len(results)} runs")
    return results


# ─────────────────────────────────────────────
# SCRIPT 8
# ─────────────────────────────────────────────

def script_8_stopping_analysis(run_name: str):
    print(f"\nSCRIPT 8 — Stopping [{run_name}]")

    df = _load_merged_with_type(run_name, ["ego_is_making_progress"])

    if "ego_is_making_progress" not in df:
        return

    df["stopped"] = df["ego_is_making_progress"] < PROGRESS_FAIL_THRESH

    print(df["stopped"].mean())

    return df


# ─────────────────────────────────────────────
# SCRIPT 9
# ─────────────────────────────────────────────

def script_9_runner_errors(run_name: str):
    print(f"\nSCRIPT 9 — Errors [{run_name}]")

    rr = load_runner_report(run_name)

    print("Columns:", rr.columns.tolist())

    err_cols = [c for c in rr.columns if "error" in c.lower()]

    for c in err_cols:
        print(c, rr[c].notna().sum())

    return rr


# ─────────────────────────────────────────────
# SCRIPT 10
# ─────────────────────────────────────────────

def script_10_html_dashboard(run_name: str):
    print(f"\nSCRIPT 10 — HTML [{run_name}]")

    out = get_output_dir() / f"dashboard_{run_name}.html"

    html = f"""
    <html>
    <body>
    <h1>Run {run_name}</h1>
    <p>Dashboard generated</p>
    </body>
    </html>
    """

    out.write_text(html)
    print("Saved:", out)

    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all_behaviour(run_name: str, compare_runs=None):
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
        run_all_behaviour(args.run, args.runs)
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
