#!/usr/bin/env python3
"""
Analysis script for scene_labels.jsonl produced by the EMoE precompute pipeline.

Generates:
1. Class Distribution
2. Tag vs Class Confusion Matrix
3. Stage Distribution
4. Travel Distance Distribution
5. "Others" Deep-Dive
6. Debug Signal Distributions
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


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

EMOE_SCENE_TYPES = [
    "left_turn", "straight_inter", "right_turn",
    "straight_non", "roundabout", "u_turn", "others"
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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_records(labels_path: Path):
    records = []
    with labels_path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def savefig(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# 1. Class Distribution
# ─────────────────────────────────────────────

def plot_class_distribution(records, out_dir):
    counts = Counter(r["emoe_class_id"] for r in records)
    total = len(records)

    vals = [counts.get(i, 0) for i in range(N_CLASSES)]
    labels = [f"[{i}] {EMOE_SCENE_TYPES[i]}" for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, vals, color=CLASS_COLORS)

    style_ax(ax, "Class Distribution", ylabel="Count")
    savefig(fig, out_dir, "1_class_distribution.png")


# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(records, out_dir, top_n_tags=25):
    pair_counts = Counter()
    tag_totals = Counter()

    for r in records:
        tag = str(r.get("scenario_type", "unknown"))
        cls = int(r["emoe_class_id"])
        pair_counts[(tag, cls)] += 1
        tag_totals[tag] += 1

    top_tags = [t for t, _ in tag_totals.most_common(top_n_tags)]

    matrix = np.zeros((len(top_tags), N_CLASSES))

    for i, tag in enumerate(top_tags):
        for j in range(N_CLASSES):
            matrix[i, j] = pair_counts.get((tag, j), 0)

    matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags)

    style_ax(ax, "Tag vs Class Confusion")

    fig.colorbar(im)
    savefig(fig, out_dir, "2_confusion.png")


# ─────────────────────────────────────────────
# 3. Stage Distribution
# ─────────────────────────────────────────────

def plot_stage_distribution(records, out_dir):
    class_stage = defaultdict(Counter)

    for r in records:
        cls = int(r["emoe_class_id"])
        stage = r.get("stage", "unknown")
        class_stage[cls][stage] += 1

    stages = list({s for c in class_stage.values() for s in c})
    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = np.zeros(N_CLASSES)

    for s in stages:
        vals = [class_stage[i].get(s, 0) for i in range(N_CLASSES)]
        ax.bar(range(N_CLASSES), vals, bottom=bottoms, label=s)
        bottoms += vals

    ax.legend(fontsize=6)
    style_ax(ax, "Stage Distribution")
    savefig(fig, out_dir, "3_stage.png")


# ─────────────────────────────────────────────
# 4. Travel Distance
# ─────────────────────────────────────────────

def plot_travel_distance(records, out_dir):
    data = defaultdict(list)

    for r in records:
        if "travel_distance_m" in r:
            data[r["emoe_class_id"]].append(r["travel_distance_m"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([data[i] for i in range(N_CLASSES)])

    style_ax(ax, "Travel Distance", ylabel="Meters")
    savefig(fig, out_dir, "4_travel_distance.png")


# ─────────────────────────────────────────────
# 5. Others Deep Dive
# ─────────────────────────────────────────────

def plot_others_deepdive(records, out_dir):
    others = [r for r in records if r["emoe_class_id"] == 6]

    tags = Counter(r.get("scenario_type", "unknown") for r in others)

    fig, ax = plt.subplots(figsize=(10, 6))
    items = tags.most_common(15)

    ax.barh([i[0] for i in items], [i[1] for i in items])

    style_ax(ax, "Others Breakdown")
    savefig(fig, out_dir, "5_others.png")


# ─────────────────────────────────────────────
# 6. Debug Signals
# ─────────────────────────────────────────────

def plot_debug_signals(records, out_dir):
    signals = ["abs_delta_heading_deg", "total_abs_heading_deg"]

    fig, axes = plt.subplots(1, len(signals), figsize=(12, 5))

    for i, sig in enumerate(signals):
        data = defaultdict(list)

        for r in records:
            val = r.get("debug", {}).get(sig)
            if val is not None:
                data[r["emoe_class_id"]].append(val)

        axes[i].boxplot([data[j] for j in range(N_CLASSES)])
        axes[i].set_title(sig)

    savefig(fig, out_dir, "6_debug.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--top_n_tags", type=int, default=25)
    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(labels_path)

    print(f"[INFO] Loaded {len(records)} records")

    plot_class_distribution(records, out_dir)
    plot_confusion_matrix(records, out_dir, args.top_n_tags)
    plot_stage_distribution(records, out_dir)
    plot_travel_distance(records, out_dir)
    plot_others_deepdive(records, out_dir)
    plot_debug_signals(records, out_dir)

    print("[DONE] All 6 plots saved.")


if __name__ == "__main__":
    main()
