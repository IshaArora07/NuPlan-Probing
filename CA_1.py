"""
analyse_core.py — Scripts 1–5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── import shared utils ─────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

# Support both load_utils and 00_load_utils
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
    load_runner_report, load_aggregator, load_metric,
    get_scenario_type_col, get_output_dir,
    set_plot_style, save_fig, bar_chart, PALETTE,
    METRIC_FILES,
)

set_plot_style()

# ─────────────────────────────────────────────
# SCRIPT 1
# ─────────────────────────────────────────────

def script_1_overview(run_name: str):
    print("\n" + "=" * 60)
    print(f"SCRIPT 1 — Overview [{run_name}]")
    print("=" * 60)

    rr = load_runner_report(run_name)
    total = len(rr)

    print(f"\nTotal scenarios: {total}")

    status_col = next((c for c in ["status", "result", "success"] if c in rr.columns), None)
    if status_col:
        counts = rr[status_col].value_counts()
        for s, c in counts.items():
            print(f"{s}: {c} ({100*c/total:.1f}%)")

    sc_col = get_scenario_type_col(rr)
    if sc_col:
        counts = rr[sc_col].value_counts()
        print("\nScenario types:")
        print(counts)

    agg = load_aggregator(run_name)
    print("\nAggregator:")
    print(agg)

    return rr, agg


# ─────────────────────────────────────────────
# SCRIPT 2
# ─────────────────────────────────────────────

def script_2_metric_deep_dive(run_name: str):
    print("\nSCRIPT 2 — Metric Analysis")

    rows = []

    for m in METRIC_FILES:
        df = load_metric(run_name, m)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        col = df[num_cols[0]].dropna()
        if len(col) == 0:
            continue

        rows.append({
            "metric": m,
            "mean": col.mean(),
            "fail_rate": (col < 0.5).mean()
        })

        del df

    summary = pd.DataFrame(rows).sort_values("fail_rate", ascending=False)
    print(summary)

    return summary


# ─────────────────────────────────────────────
# SCRIPT 3
# ─────────────────────────────────────────────

def script_3_scenario_type_breakdown(run_name: str):
    print("\nSCRIPT 3 — Scenario Type Breakdown")

    rr = load_runner_report(run_name)
    sc_col = get_scenario_type_col(rr)
    token_col = rr.columns[0]

    if sc_col is None:
        print("No scenario_type found")
        return None

    type_lookup = rr.set_index(token_col)[sc_col].to_dict()
    del rr

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for m in METRIC_FILES:
        df = load_metric(run_name, m)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        val_col = num_cols[0]

        for _, row in df[[tok, val_col]].iterrows():
            stype = type_lookup.get(row[tok])
            if stype is not None:
                grouped[m][stype].append(row[val_col])

        del df

    rows = {}
    for m, d in grouped.items():
        rows[m] = {k: np.mean(v) for k, v in d.items() if v}

    df = pd.DataFrame(rows)
    print(df)

    return df


# ─────────────────────────────────────────────
# SCRIPT 4
# ─────────────────────────────────────────────

def script_4_cross_run_comparison(runs=None):
    print("\nSCRIPT 4 — Cross Run")

    if runs is None:
        runs = list(RUN_PATHS.keys())

    rows = []

    for r in runs:
        try:
            agg = load_aggregator(r)
            row = {"run": r}
            for c in agg.columns:
                if pd.api.types.is_numeric_dtype(agg[c]):
                    row[c] = agg[c].mean()
            rows.append(row)
        except:
            continue

    df = pd.DataFrame(rows).set_index("run")
    print(df)

    return df


# ─────────────────────────────────────────────
# SCRIPT 5
# ─────────────────────────────────────────────

def script_5_failure_cases(run_name: str, top_n=30):
    print("\nSCRIPT 5 — Worst Cases")

    rr = load_runner_report(run_name)
    token_col = rr.columns[0]

    base = rr[[token_col]].copy()
    del rr

    scores = {str(t): {} for t in base[token_col]}

    for m in METRIC_FILES:
        df = load_metric(run_name, m)
        if df is None:
            continue

        tok = df.columns[0]
        num_cols = [c for c in df.columns if c != tok and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            continue

        val_col = num_cols[0]

        for _, row in df[[tok, val_col]].iterrows():
            scores[str(row[tok])][m] = row[val_col]

        del df

    for m in METRIC_FILES:
        base[m] = base[token_col].apply(lambda t: scores[str(t)].get(m, np.nan))

    base["failure_score"] = base[METRIC_FILES].apply(lambda r: (1 - r).mean(), axis=1)

    worst = base.sort_values("failure_score", ascending=False).head(top_n)
    print(worst.head())

    return worst


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

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
    else:
        print("Invalid script")
