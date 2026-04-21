#!/usr/bin/env python3
"""
Analyze EMoE scene_labels.jsonl (6-class setup).

Outputs:
1. Class distribution
2. Confusion matrix (numbers, split into multiple images)
3. Stage distribution
4. Travel distance distribution
5. "Others" deep dive
6. Debug signal distributions
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# CONFIG (6 classes ONLY)
# ─────────────────────────────────────────────

EMOE_SCENE_TYPES = [
    "left_turn",
    "straight_inter",
    "right_turn",
    "straight_non",
    "u_turn",
    "others",
]

N_CLASSES = 6


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_records(path: Path):
    records = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def savefig(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────

def plot_class_distribution(records, out_dir):
    counts = Counter(r["emoe_class_id"] for r in records)
    total = len(records)

    vals = [counts.get(i, 0) for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(N_CLASSES), vals)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(EMOE_SCENE_TYPES, rotation=25)
    ax.set_title("Class Distribution")

    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center")

    savefig(fig, out_dir, "1_class_distribution.png")


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX (FIXED)
# ─────────────────────────────────────────────

def plot_confusion_matrix_numbers(records, out_dir, top_n=25, rows_per_plot=12):
    pair_counts = Counter()
    tag_totals = Counter()

    for r in records:
        tag = str(r.get("scenario_type", "unknown"))
        cls = int(r["emoe_class_id"])
        pair_counts[(tag, cls)] += 1
        tag_totals[tag] += 1

    top_tags = [t for t, _ in tag_totals.most_common(top_n)]

    matrix = np.zeros((len(top_tags), N_CLASSES))
    for i, tag in enumerate(top_tags):
        for j in range(N_CLASSES):
            matrix[i, j] = pair_counts.get((tag, j), 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums > 0)

    num_splits = int(np.ceil(len(top_tags) / rows_per_plot))

    for split_idx in range(num_splits):
        start = split_idx * rows_per_plot
        end = min((split_idx + 1) * rows_per_plot, len(top_tags))

        sub_tags = top_tags[start:end]
        sub_matrix = matrix[start:end]

        fig, ax = plt.subplots(figsize=(12, max(6, len(sub_tags) * 0.6)))

        for i in range(len(sub_tags)):
            for j in range(N_CLASSES):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{sub_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold"
                )

        ax.set_xlim(0, N_CLASSES)
        ax.set_ylim(0, len(sub_tags))

        ax.set_xticks(np.arange(N_CLASSES) + 0.5)
        ax.set_xticklabels(EMOE_SCENE_TYPES, fontsize=12, rotation=20)

        ax.set_yticks(np.arange(len(sub_tags)) + 0.5)
        ax.set_yticklabels(sub_tags, fontsize=11)

        ax.set_title(f"Confusion Matrix Part {split_idx+1}")

        ax.invert_yaxis()

        # grid
        for i in range(len(sub_tags) + 1):
            ax.axhline(i, color="black", linewidth=0.5)
        for j in range(N_CLASSES + 1):
            ax.axvline(j, color="black", linewidth=0.5)

        savefig(fig, out_dir, f"2_confusion_matrix_part{split_idx+1}.png")


# ─────────────────────────────────────────────
# 3. STAGE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_stage_distribution(records, out_dir):
    class_stage = defaultdict(Counter)

    for r in records:
        cls = int(r["emoe_class_id"])
        stage = r.get("stage", "unknown")
        class_stage[cls][stage] += 1

    fig, ax = plt.subplots(figsize=(10, 5))

    for cls in range(N_CLASSES):
        stages = class_stage[cls]
        labels = list(stages.keys())
        values = list(stages.values())

        ax.bar([f"{cls}-{l}" for l in labels], values)

    ax.set_title("Stage Distribution")
    ax.tick_params(axis='x', rotation=45)

    savefig(fig, out_dir, "3_stage_distribution.png")


# ─────────────────────────────────────────────
# 4. TRAVEL DISTANCE
# ─────────────────────────────────────────────

def plot_travel_distance(records, out_dir):
    data = defaultdict(list)

    for r in records:
        d = r.get("travel_distance_m")
        if d is not None:
            data[int(r["emoe_class_id"])].append(d)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([data[i] for i in range(N_CLASSES)])
    ax.set_xticklabels(EMOE_SCENE_TYPES)
    ax.set_title("Travel Distance")

    savefig(fig, out_dir, "4_travel_distance.png")


# ─────────────────────────────────────────────
# 5. OTHERS DEEP DIVE
# ─────────────────────────────────────────────

def plot_others(records, out_dir):
    others = [r for r in records if r["emoe_class_id"] == 5]

    tags = Counter(r.get("scenario_type", "") for r in others)

    top = tags.most_common(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([t for t, _ in top], [c for _, c in top])
    ax.set_title("Others Breakdown")

    savefig(fig, out_dir, "5_others.png")


# ─────────────────────────────────────────────
# 6. DEBUG SIGNALS
# ─────────────────────────────────────────────

def plot_debug_signals(records, out_dir):
    signals = ["abs_delta_heading_deg", "total_abs_heading_deg"]

    data = {s: defaultdict(list) for s in signals}

    for r in records:
        cls = int(r["emoe_class_id"])
        debug = r.get("debug", {})
        for s in signals:
            if s in debug:
                data[s][cls].append(debug[s])

    fig, axes = plt.subplots(1, len(signals), figsize=(12, 5))

    for ax, s in zip(axes, signals):
        ax.boxplot([data[s][i] for i in range(N_CLASSES)])
        ax.set_title(s)
        ax.set_xticklabels(EMOE_SCENE_TYPES)

    savefig(fig, out_dir, "6_debug_signals.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading...")
    records = load_records(labels_path)
    print(f"[INFO] Loaded {len(records)} records")

    plot_class_distribution(records, out_dir)
    plot_confusion_matrix_numbers(records, out_dir)
    plot_stage_distribution(records, out_dir)
    plot_travel_distance(records, out_dir)
    plot_others(records, out_dir)
    plot_debug_signals(records, out_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
