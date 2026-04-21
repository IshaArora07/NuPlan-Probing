#!/usr/bin/env python3
"""
EMoE analysis script (6-class version, no roundabout).

Generates:
1. Class distribution
2. Tag vs class confusion matrix (NUMBERS ONLY)
3. Stage distribution
4. Travel distance distribution
5. Others (class 5) deep-dive
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
# Constants (6 classes)
# ─────────────────────────────────────────────

EMOE_SCENE_TYPES = [
    "left_turn",
    "straight_intersection",
    "right_turn",
    "straight_non_intersection",
    "u_turn",
    "others",
]

N_CLASSES = 6


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────
# 1. Class Distribution
# ─────────────────────────────────────────────

def plot_class_distribution(records, out_dir):
    counts = Counter(r["emoe_class_id"] for r in records)
    vals = [counts.get(i, 0) for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(N_CLASSES), vals)

    ax.set_title("Class Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(EMOE_SCENE_TYPES, rotation=20)

    plt.tight_layout()
    plt.savefig(out_dir / "1_class_distribution.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 2. Confusion Matrix (NUMBERS ONLY)
# ─────────────────────────────────────────────

def plot_confusion_matrix_numbers(records, out_dir, top_n=25):
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

    # Row normalize
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(top_tags) * 0.4)))

    for i in range(len(top_tags)):
        for j in range(N_CLASSES):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_xlim(0, N_CLASSES)
    ax.set_ylim(0, len(top_tags))

    ax.set_xticks(np.arange(N_CLASSES) + 0.5)
    ax.set_xticklabels(range(N_CLASSES))

    ax.set_yticks(np.arange(len(top_tags)) + 0.5)
    ax.set_yticklabels(top_tags, fontsize=8)

    ax.set_title("Tag vs Class Confusion (numbers)")
    ax.invert_yaxis()

    # grid
    for i in range(len(top_tags) + 1):
        ax.axhline(i, color="black", linewidth=0.3)
    for j in range(N_CLASSES + 1):
        ax.axvline(j, color="black", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "2_confusion_matrix_numbers.png", dpi=200)
    plt.close()


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

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(EMOE_SCENE_TYPES, rotation=20)
    ax.set_title("Stage Distribution")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "3_stage_distribution.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 4. Travel Distance
# ─────────────────────────────────────────────

def plot_travel_distance(records, out_dir):
    data = defaultdict(list)

    for r in records:
        d = r.get("travel_distance_m")
        if d is not None:
            data[int(r["emoe_class_id"])].append(d)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([data[i] for i in range(N_CLASSES)])

    ax.set_title("Travel Distance Distribution")
    ax.set_ylabel("Meters")
    ax.set_xticklabels(EMOE_SCENE_TYPES, rotation=20)

    plt.tight_layout()
    plt.savefig(out_dir / "4_travel_distance.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 5. Others Deep Dive
# ─────────────────────────────────────────────

def plot_others_deepdive(records, out_dir):
    others_class = N_CLASSES - 1  # class 5

    others = [r for r in records if int(r["emoe_class_id"]) == others_class]

    tag_counts = Counter(r.get("scenario_type", "unknown") for r in others)
    stage_counts = Counter(r.get("stage", "unknown") for r in others)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # tags
    tags = tag_counts.most_common(15)
    axes[0].barh([t[0] for t in tags], [t[1] for t in tags])
    axes[0].set_title("Others: scenario_type")

    # stages
    stages = stage_counts.most_common(10)
    axes[1].barh([s[0] for s in stages], [s[1] for s in stages])
    axes[1].set_title("Others: stages")

    plt.tight_layout()
    plt.savefig(out_dir / "5_others_deepdive.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 6. Debug Signals
# ─────────────────────────────────────────────

def plot_debug_signals(records, out_dir):
    signals = [
        "abs_delta_heading_deg",
        "total_abs_heading_deg",
        "path_len_over_dist",
    ]

    fig, axes = plt.subplots(1, len(signals), figsize=(15, 5))

    for i, sig in enumerate(signals):
        data = defaultdict(list)

        for r in records:
            val = r.get("debug", {}).get(sig)
            if val is not None:
                data[int(r["emoe_class_id"])].append(val)

        axes[i].boxplot([data[j] for j in range(N_CLASSES)])
        axes[i].set_title(sig)

    plt.tight_layout()
    plt.savefig(out_dir / "6_debug_signals.png", dpi=150)
    plt.close()


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
    plot_confusion_matrix_numbers(records, out_dir, args.top_n_tags)
    plot_stage_distribution(records, out_dir)
    plot_travel_distance(records, out_dir)
    plot_others_deepdive(records, out_dir)
    plot_debug_signals(records, out_dir)

    print("[DONE] All 6 plots saved.")


if __name__ == "__main__":
    main()
