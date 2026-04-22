"""
analyse_core.py — Scripts 1–5
EMoE Simulation Core Diagnostics
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── import shared utils ──────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from load_utils import (
    PRIMARY_RUN, RUN_PATHS,
    load_runner_report, load_aggregator, load_all_metrics, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, bar_chart, PALETTE,
    METRIC_FILES,
)

set_plot_style()

# ─────────────────────────────────────────────────────────────
# SCRIPT 1 — Overview
# ─────────────────────────────────────────────────────────────

def script_1_overview(run_name: str):
    print("\n" + "="*60)
    print(f"SCRIPT 1 — Overview & Sanity Check  [{run_name}]")
    print("="*60)

    rr = load_runner_report(run_name)
    total = len(rr)

    print(f"\nTotal scenarios: {total}")

    status_col = next((c for c in rr.columns if c.lower() in
                       ["status", "result", "success", "passed"]), None)

    if status_col:
        counts = rr[status_col].value_counts()
        for s, c in counts.items():
            print(f"  {s:<20} {c:>5} ({100*c/total:.1f}%)")

    sc_col = get_scenario_type_col(rr)
    if sc_col:
        type_counts = rr[sc_col].value_counts()

        fig, ax = plt.subplots(figsize=(9, 4))
        bar_chart(ax,
                  labels=type_counts.index.tolist(),
                  values=type_counts.values.tolist(),
                  color=PALETTE.get(run_name, "#F28C38"),
                  title=f"[{run_name}] Scenario Count by Type",
                  ylabel="Count")
        save_fig(fig, f"s1_scenario_distribution_{run_name}", subdir="core")

    agg = load_aggregator(run_name)
    print("\nAggregator columns:", list(agg.columns))

    print("\n[Script 1 complete]")
    return rr, agg


# ─────────────────────────────────────────────────────────────
# SCRIPT 2 — Metric analysis
# ─────────────────────────────────────────────────────────────

def script_2_metric_deep_dive(run_name: str):
    print("\n" + "="*60)
    print(f"SCRIPT 2 — Metric Analysis [{run_name}]")
    print("="*60)

    rows = []

    for metric_name in METRIC_FILES:
        df = load_metric(run_name, metric_name)
        if df is None:
            continue

        token_col = df.columns[0]
        num_cols = [c for c in df.columns
                    if c != token_col and pd.api.types.is_numeric_dtype(df[c])]

        if not num_cols:
            continue

        col = df[num_cols[0]].dropna()

        if len(col) == 0:
            continue

        pass_rate = (col >= 0.5).mean()
        fail_rate = (col < 0.5).mean()

        rows.append({
            "metric": metric_name,
            "mean": round(col.mean(), 4),
            "fail_rate": round(fail_rate, 4),
            "n": len(col),
        })

        del df

    summary = pd.DataFrame(rows).sort_values("fail_rate", ascending=False)

    print(summary.to_string(index=False))

    out_csv = get_output_dir() / "core" / f"s2_metric_summary_{run_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print("\n[Script 2 complete]")
    return summary


# ─────────────────────────────────────────────────────────────
# SCRIPT 3 — Scenario breakdown (memory safe)
# ─────────────────────────────────────────────────────────────

def script_3_scenario_type_breakdown(run_name: str):
    print("\n" + "="*60)
    print(f"SCRIPT 3 — Scenario Breakdown [{run_name}]")
    print("="*60)

    rr = load_runner_report(run_name)
    sc_col = get_scenario_type_col(rr)

    if sc_col is None:
        print("No scenario_type found")
        return None

    token_col = rr.columns[0]
    type_lookup = rr.set_index(token_col)[sc_col].to_dict()

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for metric_name in METRIC_FILES:
        df = load_metric(run_name, metric_name)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns
                    if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        val_col = num_cols[0]

        for _, row in df[[tok, val_col]].iterrows():
            stype = type_lookup.get(row[tok])
            if stype:
                grouped[metric_name][stype].append(row[val_col])

        del df

    rows = {}
    for metric, type_vals in grouped.items():
        rows[metric] = {t: np.mean(v) for t, v in type_vals.items()}

    df_out = pd.DataFrame(rows)

    print(df_out.to_string())

    print("\n[Script 3 complete]")
    return df_out


# ─────────────────────────────────────────────────────────────
# SCRIPT 4 — Cross-run
# ─────────────────────────────────────────────────────────────

def script_4_cross_run_comparison(runs=None):
    print("\n" + "="*60)
    print("SCRIPT 4 — Cross-run")
    print("="*60)

    runs = runs or list(RUN_PATHS.keys())

    rows = []

    for run in runs:
        try:
            agg = load_aggregator(run)
            val = agg.select_dtypes("number").mean().mean()
            rows.append({"run": run, "score": val})
        except:
            continue

    df = pd.DataFrame(rows)
    print(df)

    print("\n[Script 4 complete]")
    return df


# ─────────────────────────────────────────────────────────────
# SCRIPT 5 — Worst cases
# ─────────────────────────────────────────────────────────────

def script_5_failure_cases(run_name: str, top_n=30):
    print("\n" + "="*60)
    print(f"SCRIPT 5 — Worst cases [{run_name}]")
    print("="*60)

    merged = load_all_metrics(run_name)
    token_col = merged.columns[0]

    metric_cols = [c for c in merged.columns if c != token_col]

    merged["failure_score"] = merged[metric_cols].apply(
        lambda r: (1 - r).mean(), axis=1
    )

    worst = merged.sort_values("failure_score", ascending=False).head(top_n)

    print(worst[[token_col, "failure_score"]].head())

    print("\n[Script 5 complete]")
    return worst


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all_core(run_name: str, compare_runs=None):
    script_1_overview(run_name)
    script_2_metric_deep_dive(run_name)
    script_3_scenario_type_breakdown(run_name)
    script_4_cross_run_comparison(compare_runs)
    script_5_failure_cases(run_name)


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
