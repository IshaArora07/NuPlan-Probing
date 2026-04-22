#!/usr/bin/env python3
"""
analyse_core.py — Scripts 1–5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FIX
sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_aggregator, load_all_metrics, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, bar_chart, PALETTE,
    METRIC_FILES,
)

set_plot_style()

# ─────────────────────────────────────────────
# SCRIPT 1
# ─────────────────────────────────────────────

def script_1_overview(run_name):
    print("\n" + "="*60)
    print(f"SCRIPT 1 — Overview [{run_name}]")
    print("="*60)

    rr = load_runner_report(run_name)
    total = len(rr)

    print(f"\nTotal scenarios: {total}")

    status_col = next((c for c in ["status","result","success"] if c in rr.columns), None)

    if status_col:
        counts = rr[status_col].value_counts()
        for k,v in counts.items():
            print(f"{k:<15} {v} ({100*v/total:.1f}%)")

    sc_col = get_scenario_type_col(rr)
    if sc_col:
        type_counts = rr[sc_col].value_counts()
        print("\nScenario types:")
        print(type_counts.to_string())

    agg = load_aggregator(run_name)
    print("\nAggregator:")
    print(agg.to_string())

    return rr, agg


# ─────────────────────────────────────────────
# SCRIPT 2
# ─────────────────────────────────────────────

def script_2_metric_deep_dive(run_name):
    print("\n" + "="*60)
    print(f"SCRIPT 2 — Metric Analysis [{run_name}]")
    print("="*60)

    rows = []

    for m in METRIC_FILES:
        df = load_metric(run_name, m)
        if df is None:
            continue

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        col = df[num_cols[0]].dropna()
        if len(col) == 0:
            continue

        is_binary = set(col.unique()).issubset({0,1})

        if is_binary:
            pass_rate = col.mean()
            fail_rate = 1-pass_rate
        else:
            pass_rate = (col>=0.5).mean()
            fail_rate = (col<0.5).mean()

        rows.append({
            "metric": m,
            "mean": col.mean(),
            "fail_rate": fail_rate
        })

    df = pd.DataFrame(rows).sort_values("fail_rate", ascending=False)
    print(df.to_string(index=False))

    return df


# ─────────────────────────────────────────────
# SCRIPT 3
# ─────────────────────────────────────────────

def script_3_scenario_type_breakdown(run_name):
    print("\n" + "="*60)
    print(f"SCRIPT 3 — Scenario Breakdown [{run_name}]")
    print("="*60)

    merged = load_all_metrics(run_name)
    rr = load_runner_report(run_name)

    token = merged.columns[0]

    if "scenario_type" in rr.columns:
        merged = pd.merge(merged, rr[[token,"scenario_type"]], on=token, how="left")

    if "scenario_type" not in merged.columns:
        print("No scenario_type found")
        return

    grouped = merged.groupby("scenario_type").mean()

    print(grouped.round(3).to_string())

    return merged, grouped


# ─────────────────────────────────────────────
# SCRIPT 4
# ─────────────────────────────────────────────

def script_4_cross_run_comparison(runs=None):
    print("\n" + "="*60)
    print("SCRIPT 4 — Cross Run")
    print("="*60)

    if runs is None:
        runs = list(RUN_PATHS.keys())

    rows = []

    for r in runs:
        try:
            merged = load_all_metrics(r)
            token = merged.columns[0]

            row = {"run": r}
            for c in merged.columns:
                if c != token:
                    row[c] = merged[c].mean()

            rows.append(row)

        except:
            print(f"Skip {r}")

    df = pd.DataFrame(rows).set_index("run")

    print(df.round(3).to_string())

    return df


# ─────────────────────────────────────────────
# SCRIPT 5
# ─────────────────────────────────────────────

def script_5_failure_cases(run_name, top_n=30):
    print("\n" + "="*60)
    print(f"SCRIPT 5 — Worst Cases [{run_name}]")
    print("="*60)

    merged = load_all_metrics(run_name)

    token = merged.columns[0]
    metrics = [c for c in merged.columns if c != token]

    merged["failure"] = merged[metrics].apply(lambda r: (1-r).mean(), axis=1)

    worst = merged.sort_values("failure", ascending=False).head(top_n)

    print(worst[[token,"failure"]].to_string(index=False))

    return worst


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all_core(run):
    script_1_overview(run)
    script_2_metric_deep_dive(run)
    script_3_scenario_type_breakdown(run)
    script_4_cross_run_comparison()
    script_5_failure_cases(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=PRIMARY_RUN)
    parser.add_argument("--script", type=int, default=0)

    args = parser.parse_args()

    if args.script == 0:
        run_all_core(args.run)
    elif args.script == 1:
        script_1_overview(args.run)
    elif args.script == 2:
        script_2_metric_deep_dive(args.run)
    elif args.script == 3:
        script_3_scenario_type_breakdown(args.run)
    elif args.script == 4:
        script_4_cross_run_comparison()
    elif args.script == 5:
        script_5_failure_cases(args.run)
