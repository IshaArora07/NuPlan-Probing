#!/usr/bin/env python3
"""
analyse_core.py  —  Scripts 1–5
EMoE Simulation Core Diagnostics

Scripts:
1. Overview & sanity check
2. Per-metric failure analysis
3. Scenario-type breakdown
4. Cross-run comparison (T2 / T3 / T4 …)
5. Worst-case failure export

Usage:
python analyse_core.py                  # runs all 5 on PRIMARY_RUN
python analyse_core.py --run T4         # override primary run
python analyse_core.py --script 3       # run only script 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # kept because you asked not to remove comments

# ── import shared utils ──────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from load_utils import (  # noqa: E402
    PRIMARY_RUN,
    RUN_PATHS,
    load_runner_report,
    load_aggregator,
    load_all_metrics,
    load_metric,
    get_scenario_type_col,
    get_output_dir,
    set_plot_style,
    save_fig,
    bar_chart,
    PALETTE,
    METRIC_FILES,
)

set_plot_style()

# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 1 — Overview & sanity check
# ─────────────────────────────────────────────────────────────────────────────


def script_1_overview(run_name: str):
    print("\n" + "=" * 60)
    print(f"SCRIPT 1 — Overview & Sanity Check  [{run_name}]")
    print("=" * 60)

    # ── Runner report ────────────────────────────────────────────
    rr = load_runner_report(run_name)

    total = len(rr)
    print(f"\nTotal scenarios in runner report : {total}")

    # Status column detection
    status_col = next(
        (
            c
            for c in ["status", "Status", "result", "Result", "success", "passed"]
            if c in rr.columns
        ),
        None,
    )
    if status_col:
        counts = rr[status_col].value_counts()
        print(f"\nScenario status breakdown ({status_col}):")
        for status, cnt in counts.items():
            pct = 100 * cnt / total
            print(f"  {str(status):<20} {cnt:>5}  ({pct:.1f}%)")
    else:
        print("  [INFO] No status/result column found in runner_report.")
        print(f"  Columns available: {list(rr.columns)}")

    # Scenario type distribution
    sc_col = get_scenario_type_col(rr)
    if sc_col:
        type_counts = rr[sc_col].value_counts()
        print(f"\nScenario type distribution ({sc_col}):")
        for stype, cnt in type_counts.items():
            pct = 100 * cnt / total
            print(f"  {str(stype):<35} {cnt:>5}  ({pct:.1f}%)")
    else:
        print("\n  [INFO] No scenario_type column auto-detected in runner_report.")

    # ── Aggregator metric ────────────────────────────────────────
    agg = load_aggregator(run_name)
    print(f"\nAggregator metric columns: {list(agg.columns)}")

    # Try to find the overall weighted score
    score_col = next(
        (c for c in agg.columns if "score" in c.lower() or "weighted" in c.lower()),
        None,
    )
    if score_col:
        val = agg[score_col].values
        if len(val) == 1:
            print(f"\n  Overall weighted score ({score_col}): {val[0]:.4f}")
        else:
            print(f"\n  {score_col}:")
            print(agg[[score_col]].to_string(index=False))
    else:
        print("\n  Full aggregator table:")
        print(agg.to_string())

    # ── Error / crashed scenarios ────────────────────────────────
    err_cols = [
        c
        for c in rr.columns
        if any(k in c.lower() for k in ["error", "exception", "crash", "timeout", "skip"])
    ]
    if err_cols:
        for ec in err_cols:
            if rr[ec].dtype == object:
                n_err = rr[ec].notna().sum()
            else:
                n_err = (rr[ec] > 0).sum()
            print(f"\n  Errored scenarios ({ec}): {n_err}")
    else:
        print("\n  [INFO] No explicit error columns found — check log.txt manually.")

    # ── Plot: scenario type bar chart ────────────────────────────
    if sc_col and len(type_counts) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        bar_chart(
            ax,
            labels=type_counts.index.tolist(),
            values=type_counts.values.tolist(),
            color=PALETTE.get(run_name, "#F28C38"),
            title=f"[{run_name}] Scenario Count by Type",
            ylabel="Count",
        )
        save_fig(fig, f"s1_scenario_distribution_{run_name}", subdir="core")

    print("\n[Script 1 complete]")
    return rr, agg


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 2 — Per-metric failure analysis
# ─────────────────────────────────────────────────────────────────────────────


def script_2_metric_deep_dive(run_name: str):
    print("\n" + "=" * 60)
    print(f"SCRIPT 2 — Per-Metric Failure Analysis  [{run_name}]")
    print("=" * 60)

    merged = load_all_metrics(run_name)

    # For each metric, compute mean, std, min, % below 0.5 (failure proxy)
    token_col = merged.columns[0]
    metric_cols = [c for c in merged.columns if c != token_col]

    rows = []
    for mc in metric_cols:
        col = merged[mc].dropna()
        if len(col) == 0:
            continue

        # Binary metrics (0/1) vs continuous
        is_binary = col.isin([0, 1]).all()
        if is_binary:
            pass_rate = col.mean()
            fail_rate = 1 - pass_rate
            mean_val = pass_rate
            std_val = col.std()
        else:
            pass_rate = (col >= 0.5).mean()
            fail_rate = (col < 0.5).mean()
            mean_val = col.mean()
            std_val = col.std()

        rows.append(
            {
                "metric": mc,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(col.min(), 4),
                "pass_rate": round(pass_rate, 4),
                "fail_rate": round(fail_rate, 4),
                "n_scenarios": len(col),
            }
        )

    summary = pd.DataFrame(rows).sort_values("fail_rate", ascending=False)
    print("\nMetric failure rates (sorted worst → best):")
    print(summary.to_string(index=False))

    # Save CSV
    out_csv = get_output_dir() / "core" / f"s2_metric_summary_{run_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\n  Saved CSV → {out_csv}")

    # ── Plot: fail rate bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "#FF6B6B" if fr > 0.3 else "#F28C38" if fr > 0.1 else "#6BCB77"
        for fr in summary["fail_rate"]
    ]
    bars = ax.bar(
        summary["metric"],
        summary["fail_rate"],
        color=colors,
        width=0.6,
        zorder=3,
    )
    ax.set_title(f"[{run_name}] Metric Failure Rates", fontsize=12, fontweight="bold")
    ax.set_ylabel("Failure Rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.1, color="#888", linestyle="--", linewidth=0.8, label="10% threshold")
    ax.axhline(0.3, color="#FF6B6B", linestyle="--", linewidth=0.8, label="30% threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, summary["fail_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#FFFFFF",
        )
    save_fig(fig, f"s2_metric_fail_rates_{run_name}", subdir="core")

    print("\n[Script 2 complete]")
    return merged, summary


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 3 — Scenario-type breakdown
# ─────────────────────────────────────────────────────────────────────────────


def script_3_scenario_type_breakdown(run_name: str, merged: pd.DataFrame = None):
    print("\n" + "=" * 60)
    print(f"SCRIPT 3 — Scenario Type Breakdown  [{run_name}]")
    print("=" * 60)

    if merged is None:
        merged = load_all_metrics(run_name)

    # We need scenario_type — try merging from runner_report
    sc_col = get_scenario_type_col(merged)
    if sc_col is None:
        rr = load_runner_report(run_name)
        token_col_merged = merged.columns[0]
        token_col_rr = token_col_merged if token_col_merged in rr.columns else None
        rr_sc_col = get_scenario_type_col(rr)

        if token_col_rr and rr_sc_col:
            merged = pd.merge(
                merged,
                rr[[token_col_rr, rr_sc_col]].rename(columns={rr_sc_col: "scenario_type"}),
                on=token_col_rr,
                how="left",
            )
            sc_col = "scenario_type"
        else:
            print("  [WARN] Cannot identify scenario_type column. Skipping type breakdown.")
            print("  → Check column names in runner_report manually.")
            return merged

    token_col = merged.columns[0]
    metric_cols = [c for c in merged.columns if c not in [token_col, sc_col]]

    # Per scenario type: mean of each metric
    grouped = merged.groupby(sc_col)[metric_cols].mean().round(4)
    print("\nMean metric scores by scenario type:")
    print(grouped.to_string())

    out_csv = get_output_dir() / "core" / f"s3_by_scenario_type_{run_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_csv)
    print(f"\n  Saved CSV → {out_csv}")

    # ── Heatmap ─────────────────────────────────────────────────
    # Pick the most diagnostic metrics to keep heatmap readable
    key_metrics = [
        m
        for m in [
            "ego_is_comfortable",
            "ego_is_making_progress",
            "no_ego_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "speed_limit_compliance",
            "driving_direction_compliance",
            "ego_progress_along_expert_route",
        ]
        if m in grouped.columns
    ]

    if len(key_metrics) > 0:
        heat_data = grouped[key_metrics]
        fig, ax = plt.subplots(
            figsize=(max(10, len(key_metrics) * 1.2), max(4, len(grouped) * 0.7))
        )
        cmap = plt.cm.RdYlGn
        im = ax.imshow(heat_data.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(key_metrics)))
        ax.set_xticklabels(
            [m.replace("ego_", "").replace("_", "\n") for m in key_metrics],
            fontsize=8,
        )
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped.index.tolist(), fontsize=8)
        ax.set_title(
            f"[{run_name}] Mean Metric Score by Scenario Type",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Score (0→1)")
        # Annotate cells
        for i in range(len(grouped)):
            for j in range(len(key_metrics)):
                val = heat_data.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if 0.3 < val < 0.8 else "white",
                    )
        save_fig(fig, f"s3_heatmap_{run_name}", subdir="core")

    # ── Per-type score bar chart for key binary metrics ──────────
    binary_metrics = [
        "ego_is_comfortable",
        "ego_is_making_progress",
        "no_ego_at_fault_collisions",
        "drivable_area_compliance",
    ]
    binary_metrics = [m for m in binary_metrics if m in grouped.columns]

    if binary_metrics:
        fig, axes = plt.subplots(
            1,
            len(binary_metrics),
            figsize=(4 * len(binary_metrics), 4),
            sharey=False,
        )
        if len(binary_metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, binary_metrics):
            vals = grouped[metric].values
            types = grouped.index.tolist()
            colors = [
                "#FF6B6B" if v < 0.7 else "#F28C38" if v < 0.9 else "#6BCB77"
                for v in vals
            ]
            ax.bar(range(len(types)), vals, color=colors, zorder=3)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(
                [str(t).replace("_", "\n") for t in types],
                fontsize=7,
                rotation=0,
            )
            ax.set_ylim(0, 1.1)
            ax.set_title(metric.replace("_", " "), fontsize=9, fontweight="bold")
            ax.axhline(1.0, color="#888", linewidth=0.6, linestyle="--")
            ax.grid(axis="y", zorder=0)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color="#FFFFFF")
        fig.suptitle(
            f"[{run_name}] Key Metric Pass Rates by Scenario Type",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        save_fig(fig, f"s3_per_type_bars_{run_name}", subdir="core")

    print("\n[Script 3 complete]")
    return merged, grouped


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 4 — Cross-run comparison
# ─────────────────────────────────────────────────────────────────────────────


def script_4_cross_run_comparison(runs: list = None):
    print("\n" + "=" * 60)
    print("SCRIPT 4 — Cross-Run Comparison")
    print("=" * 60)

    if runs is None:
        runs = list(RUN_PATHS.keys())

    # Load aggregator for each available run
    agg_rows = []
    metric_rows = []

    for run in runs:
        run_p = Path(RUN_PATHS[run])
        if not run_p.exists():
            print(f"  [SKIP] {run} — path not found: {run_p}")
            continue

        # Aggregator
        try:
            agg = load_aggregator(run)
            row = {"run": run}
            for col in agg.columns:
                if pd.api.types.is_numeric_dtype(agg[col]):
                    row[col] = agg[col].values[0] if len(agg) == 1 else agg[col].mean()
            agg_rows.append(row)
        except Exception as e:
            print(f"  [WARN] Could not load aggregator for {run}: {e}")

        # Per-metric means
        try:
            merged = load_all_metrics(run)
            token_col = merged.columns[0]
            mrow = {"run": run}
            for mc in [c for c in merged.columns if c != token_col]:
                mrow[mc] = merged[mc].mean()
            metric_rows.append(mrow)
        except Exception as e:
            print(f"  [WARN] Could not load metrics for {run}: {e}")

    if not agg_rows and not metric_rows:
        print("  No runs available for comparison.")
        return None, None

    agg_df = None
    metric_df = None

    # ── Aggregator comparison ────────────────────────────────────
    if agg_rows:
        agg_df = pd.DataFrame(agg_rows).set_index("run")
        print("\nAggregator scores across runs:")
        print(agg_df.to_string())

        out_csv = get_output_dir() / "core" / "s4_aggregator_comparison.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(out_csv)
        print(f"\n  Saved → {out_csv}")

        # Bar chart per aggregator metric
        num_cols = [c for c in agg_df.columns if pd.api.types.is_numeric_dtype(agg_df[c])]
        if num_cols:
            fig, axes = plt.subplots(1, len(num_cols), figsize=(max(6, 3 * len(num_cols)), 4))
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                vals = agg_df[col].values
                labels = agg_df.index.tolist()
                colors = [PALETTE.get(r, "#888") for r in labels]
                ax.bar(labels, vals, color=colors, zorder=3)
                ax.set_title(col.replace("_", "\n"), fontsize=8, fontweight="bold")
                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", zorder=0)
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color="#FFFFFF")
            fig.suptitle(
                "Cross-Run Aggregator Score Comparison",
                fontsize=11,
                fontweight="bold",
            )
            plt.tight_layout()
            save_fig(fig, "s4_aggregator_comparison", subdir="core")

    # ── Per-metric comparison ────────────────────────────────────
    if metric_rows:
        metric_df = pd.DataFrame(metric_rows).set_index("run")
        print("\nPer-metric mean scores across runs:")
        print(metric_df.round(4).to_string())

        out_csv2 = get_output_dir() / "core" / "s4_metric_comparison.csv"
        out_csv2.parent.mkdir(parents=True, exist_ok=True)
        metric_df.to_csv(out_csv2)
        print(f"\n  Saved → {out_csv2}")

        # Grouped bar chart — key metrics only
        key = [
            m
            for m in [
                "ego_is_comfortable",
                "ego_is_making_progress",
                "no_ego_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "speed_limit_compliance",
            ]
            if m in metric_df.columns
        ]

        if key:
            x = np.arange(len(key))
            width = 0.8 / len(metric_df)
            fig, ax = plt.subplots(figsize=(12, 5))
            for i, (run, row) in enumerate(metric_df[key].iterrows()):
                offset = (i - len(metric_df) / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    row.values,
                    width,
                    label=run,
                    color=PALETTE.get(run, "#888"),
                    zorder=3,
                    alpha=0.9,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [m.replace("ego_", "").replace("_", "\n") for m in key],
                fontsize=8,
            )
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Mean Score")
            ax.set_title("Cross-Run Key Metric Comparison", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", zorder=0)
            save_fig(fig, "s4_metric_comparison_grouped", subdir="core")

    print("\n[Script 4 complete]")
    return agg_df, metric_df


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT 5 — Worst-case failure export
# ─────────────────────────────────────────────────────────────────────────────


def script_5_failure_cases(run_name: str, merged: pd.DataFrame = None, top_n: int = 30):
    print("\n" + "=" * 60)
    print(f"SCRIPT 5 — Worst-Case Failure Export  [{run_name}]  (top {top_n})")
    print("=" * 60)

    if merged is None:
        merged = load_all_metrics(run_name)

    token_col = merged.columns[0]
    metric_cols = [c for c in merged.columns if c != token_col and c != "scenario_type"]

    # Composite failure score: mean of (1 - metric) across available columns
    # So higher = worse
    score_df = merged[metric_cols].copy()
    merged = merged.copy()
    merged["failure_score"] = score_df.apply(lambda row: (1 - row.dropna()).mean(), axis=1)

    worst = merged.sort_values("failure_score", ascending=False).head(top_n)

    print(f"\nTop {top_n} worst scenarios by composite failure score:")
    display_cols = [token_col] + (["scenario_type"] if "scenario_type" in worst.columns else []) + [
        "failure_score"
    ] + metric_cols[:6]
    print(worst[display_cols].to_string(index=False))

    out_csv = get_output_dir() / "core" / f"s5_worst_scenarios_{run_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    worst.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}")
    print("  → Use token values to load these in nuBoard for visual inspection")

    # ── Plot: failure score distribution ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of failure scores
    ax = axes[0]
    ax.hist(
        merged["failure_score"].dropna(),
        bins=40,
        color=PALETTE.get(run_name, "#F28C38"),
        edgecolor="#0F1117",
        zorder=3,
    )
    ax.set_xlabel("Composite Failure Score")
    ax.set_ylabel("# Scenarios")
    ax.set_title(f"[{run_name}] Failure Score Distribution", fontweight="bold")
    ax.axvline(
        merged["failure_score"].quantile(0.9),
        color="#FF6B6B",
        linestyle="--",
        linewidth=1,
        label="90th percentile",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", zorder=0)

    # Radar / bar of worst scenario's metric profile (top 1)
    ax2 = axes[1]
    worst1 = worst.iloc[0]
    vals = [worst1[m] for m in metric_cols if m in worst1.index]
    names = [m for m in metric_cols if m in worst1.index]
    colors = [
        "#FF6B6B" if v < 0.5 else "#F28C38" if v < 0.8 else "#6BCB77"
        for v in vals
    ]
    ax2.barh(range(len(names)), vals, color=colors, zorder=3)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(
        [n.replace("ego_", "").replace("_", " ") for n in names],
        fontsize=7,
    )
    ax2.set_xlim(0, 1.1)
    ax2.axvline(0.5, color="#888", linestyle="--", linewidth=0.8)
    ax2.set_title(
        f"Worst Scenario Profile\n{str(worst1[token_col])[:40]}",
        fontsize=9,
        fontweight="bold",
    )
    ax2.grid(axis="x", zorder=0)

    plt.tight_layout()
    save_fig(fig, f"s5_failure_analysis_{run_name}", subdir="core")

    # ── Per-metric count of how many worst-N scenarios failed ───
    print(f"\nMetric failure counts in worst {top_n} scenarios:")
    for mc in metric_cols:
        n_fail = (worst[mc] < 0.5).sum()
        print(f"  {mc:<45} {n_fail:>3} / {top_n}")

    print("\n[Script 5 complete]")
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def run_all_core(run_name: str, compare_runs: list = None):
    print(f"\n{'#' * 60}")
    print(f"  analyse_core.py  —  Run: {run_name}")
    print(f"{'#' * 60}")

    rr, agg = script_1_overview(run_name)
    merged, summary = script_2_metric_deep_dive(run_name)
    merged, grouped = script_3_scenario_type_breakdown(run_name, merged)
    script_4_cross_run_comparison(compare_runs)
    script_5_failure_cases(run_name, merged)

    print(f"\n{'#' * 60}")
    print(f"  All core scripts complete. Outputs in: {get_output_dir() / 'core'}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMoE Core Analysis (Scripts 1-5)")
    parser.add_argument(
        "--run",
        default=PRIMARY_RUN,
        help="Primary run name (default: PRIMARY_RUN from config)",
    )
    parser.add_argument(
        "--script",
        type=int,
        default=0,
        help="Run only this script number (1-5). 0 = all.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of worst scenarios to export (Script 5)",
    )
    args = parser.parse_args()

    run = args.run

    if args.script == 0:
        run_all_core(run)
    elif args.script == 1:
        script_1_overview(run)
    elif args.script == 2:
        script_2_metric_deep_dive(run)
    elif args.script == 3:
        script_3_scenario_type_breakdown(run)
    elif args.script == 4:
        script_4_cross_run_comparison()
    elif args.script == 5:
        merged = load_all_metrics(run)
        script_5_failure_cases(run, merged, top_n=args.top_n)
    else:
        print(f"Invalid script number: {args.script}. Choose 1-5 or 0 for all.")
