#!/usr/bin/env python3
"""
Analyze + visualize how many scenarios whose scenario_type contains "pickup_dropoff"
are being classified into each EMoE class (and which stage did it).

Uses ONLY your existing scene_labels.jsonl (no re-classification).

Run:
  python analyze_pickup_dropoff.py \
    --scene_labels /path/to/scene_labels.jsonl \
    --out_dir ./pickup_dropoff_analysis \
    --pattern pickup_dropoff
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_labels", type=str, required=True, help="Path to scene_labels.jsonl")
    parser.add_argument("--out_dir", type=str, default="./pickup_dropoff_analysis", help="Where to save outputs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="pickup_dropoff",
        help="Substring to match in scenario_type (case-insensitive)",
    )
    args = parser.parse_args()

    labels_path = Path(args.scene_labels).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern.lower().strip()

    total_rows = 0
    matched_rows = 0

    class_counts = Counter()          # "cid:cname" -> count
    stage_counts = Counter()          # stage -> count
    class_to_stage = defaultdict(Counter)  # "cid:cname" -> (stage -> count)

    with labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            stype = str(obj.get("scenario_type", "") or "")
            if pattern not in stype.lower():
                continue

            matched_rows += 1
            cid = int(obj.get("emoe_class_id", -1))
            cname = str(obj.get("emoe_class_name", "unknown"))
            stage = str(obj.get("stage", "unknown"))

            class_key = f"{cid}:{cname}"
            class_counts[class_key] += 1
            stage_counts[stage] += 1
            class_to_stage[class_key][stage] += 1

    # ---------- Save text summary ----------
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as w:
        w.write(f"scene_labels: {labels_path}\n")
        w.write(f"pattern: {pattern}\n")
        w.write(f"total rows scanned: {total_rows}\n")
        w.write(f"matched rows: {matched_rows}\n\n")

        w.write("Class distribution (matched):\n")
        for k, v in class_counts.most_common():
            w.write(f"  {k:32s}  {v}\n")

        w.write("\nStage distribution (matched):\n")
        for k, v in stage_counts.most_common():
            w.write(f"  {k:40s}  {v}\n")

        w.write("\nPer-class stage breakdown (matched):\n")
        for ck, ctr in class_to_stage.items():
            w.write(f"\n  {ck}\n")
            for sk, sv in ctr.most_common():
                w.write(f"    {sk:40s}  {sv}\n")

    # ---------- Plots ----------
    if matched_rows > 0:
        # Class distribution plot
        class_labels = [k for k, _ in class_counts.most_common()]
        class_values = [v for _, v in class_counts.most_common()]

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(class_labels)), class_values)
        plt.xticks(range(len(class_labels)), class_labels, rotation=30, ha="right")
        plt.ylabel("Count")
        plt.title(f"EMoE class distribution for scenario_type containing '{pattern}' (n={matched_rows})")
        plt.tight_layout()
        plt.savefig(out_dir / "class_distribution.png", dpi=160)
        plt.close()

        # Stage distribution plot
        stage_labels = [k for k, _ in stage_counts.most_common()]
        stage_values = [v for _, v in stage_counts.most_common()]

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(stage_labels)), stage_values)
        plt.xticks(range(len(stage_labels)), stage_labels, rotation=30, ha="right")
        plt.ylabel("Count")
        plt.title(f"Stage distribution for scenario_type containing '{pattern}' (n={matched_rows})")
        plt.tight_layout()
        plt.savefig(out_dir / "stage_distribution.png", dpi=160)
        plt.close()

    print(f"[DONE] Scanned {total_rows} rows, matched {matched_rows}.")
    print(f"[OUT]  {out_dir}")
    print(f"[OUT]  {summary_path}")
    if matched_rows > 0:
        print(f"[OUT]  {out_dir / 'class_distribution.png'}")
        print(f"[OUT]  {out_dir / 'stage_distribution.png'}")
    else:
        print("[WARN] No matched rows; plots not created.")


if __name__ == "__main__":
    main()
