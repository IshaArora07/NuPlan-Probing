#!/usr/bin/env python3
"""
Plot EMOE class distribution from scene_labels.jsonl

Outputs:
  - class_distribution.png
"""

import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
    "others",                         # 6
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to scene_labels.jsonl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="class_distribution.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    counts = Counter()

    print("[INFO] Reading labels...")
    with open(args.labels, "r") as f:
        for line in f:
            record = json.loads(line)
            cls_id = int(record["emoe_class_id"])
            counts[cls_id] += 1

    print("\n[INFO] Class counts:")
    for cid, name in enumerate(EMOE_SCENE_TYPES):
        print(f"{name:28s}: {counts[cid]}")

    # Prepare plot data
    labels = EMOE_SCENE_TYPES
    values = [counts[i] for i in range(len(EMOE_SCENE_TYPES))]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of scenarios")
    plt.title("EMOE Scene Class Distribution")
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    print(f"\n[INFO] Saved plot to {args.out}")


if __name__ == "__main__":
    main()
