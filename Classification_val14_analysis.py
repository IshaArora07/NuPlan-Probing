#!/usr/bin/env python3
"""
Analysis script for scene_labels.jsonl produced by the EMoE precompute pipeline.

Analyses performed:

1. Class Distribution         — scenario counts per EMoE class
1. Tag vs Class Confusion     — nuPlan scenario_type  x  emoe_class_id heatmap
1. Stage Distribution         — which classifier stage fired, per class
1. Travel Distance            — boxplot of travel_distance_m per class
1. “Others” Deep-Dive         — scenario_type breakdown + stage breakdown for class 6
1. Debug Signal Distributions — abs_delta_heading_deg, total_abs_heading_deg,
   path_len_over_dist per class (violin / histogram)

Usage:
python analyze_scene_labels.py \
  --labels_path ./emoe_precomputed/scene_labels.jsonl \
  --output_dir  ./analysis_plots

All plots are saved as PNG to --output_dir. A summary table is printed to stdout.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

EMOE_SCENE_TYPES = [
    "left_turn",
    "straight_inter",
    "right_turn",
    "straight_non",
    "roundabout",
    "u_turn",
    "others",
]

EMOE_FULL_NAMES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]

N_CLASSES = 7

CLASS_COLORS = [
    "#4C72B0", "#55A868", "#C44E52",
    "#8172B2", "#CCB974", "#64B5CD", "#A9A9A9",
]

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_records(labels_path: Path):
    records = []
    with labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def savefig(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

# ──────────────────────────────────────────────────────────────
# 1. Class Distribution
# ──────────────────────────────────────────────────────────────

def plot_class_distribution(records, out_dir):
    print("[1] Class distribution...")
    counts = Counter(r["emoe_class_id"] for r in records)
    total = len(records)

    ids = list(range(N_CLASSES))
    vals = [counts.get(i, 0) for i in ids]
    labels = [f"[{i}] {EMOE_SCENE_TYPES[i]}" for i in ids]
    pcts = [100.0 * v / max(1, total) for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=CLASS_COLORS, edgecolor="white", linewidth=0.8)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    style_ax(ax, "EMoE Class Distribution", ylabel="Scenario Count")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    savefig(fig, out_dir, "1_class_distribution.png")

# ──────────────────────────────────────────────────────────────
# 6. Debug Signals (only section shown shortened fix context)
# ──────────────────────────────────────────────────────────────

def plot_debug_signals(records, out_dir):
    print("\n[6] Debug signal distributions per class...")

    signals = {
        "abs_delta_heading_deg": "Abs Net Heading Change (°)",
        "total_abs_heading_deg": "Total Abs Heading (°)",
        "path_len_over_dist":    "Path Length / Straight Dist (loopiness)",
    }

    data: Dict = {sig: defaultdict(list) for sig in signals}

    for r in records:
        cls = int(r["emoe_class_id"])
        debug = r.get("debug", {}) or {}

        for sig in signals:
            val = debug.get(sig, None)
            if val is not None:
                try:
                    data[sig][cls].append(float(val))
                except (TypeError, ValueError):
                    pass

    fig, axes = plt.subplots(1, len(signals), figsize=(18, 6))

    for ax, (sig, ylabel) in zip(axes, signals.items()):
        class_data = [data[sig].get(i, []) for i in range(N_CLASSES)]

        all_vals = [v for d in class_data for v in d]
        if not all_vals:
            ax.set_title(ylabel)
            continue

        p95 = float(np.percentile(all_vals, 95))

        vp = ax.violinplot(
            [np.clip(d, 0, p95 * 1.5) if d else [0] for d in class_data],
            positions=range(N_CLASSES),
            showmedians=True,
            showextrema=False,
            widths=0.7,
        )

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(CLASS_COLORS[i])
            body.set_alpha(0.75)
            body.set_edgecolor("white")

        vp["cmedians"].set_color("#111")
        vp["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(
            [f"[{i}]\n{EMOE_SCENE_TYPES[i]}" for i in range(N_CLASSES)],
            fontsize=8, rotation=20, ha="right")

        style_ax(ax, ylabel, ylabel=ylabel)

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to scene_labels.jsonl"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same folder as labels_path)"
    )

    parser.add_argument(
        "--top_n_tags",
        type=int,
        default=25,
        help="Number of top scenario_types to show"
    )

    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading records from: {labels_path}")
    records = load_records(labels_path)
    print(f"[INFO] Loaded {len(records):,} records")

    plot_class_distribution(records, out_dir)
    plot_debug_signals(records, out_dir)

    print("\n[DONE] All plots saved.")


if __name__ == "__main__":
    main()
