#!/usr/bin/env python3
"""
Find misclassified tokens for a given class — trajectories whose endpoint
y is on the wrong side — and print their classifier stage + debug info.

For left turn  (class 0): wrong = y < 0
For right turn (class 2): wrong = y > 0

Usage:
python find_misclassified.py \
--cache_dir   ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--class_id    0

python find_misclassified.py \
--cache_dir   ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--class_id    2
"""

import gzip
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]

# for each class, which sign of y is wrong
WRONG_SIDE = {
    0: lambda y: y < 0,   # left turn should be y > 0
    2: lambda y: y > 0,   # right turn should be y < 0
}


def load_emoe_class(feat_path: Path):
    try:
        raw = pickle.load(gzip.open(feat_path, "rb"))
        inner = raw["data"]

        if hasattr(inner, "data"):
            inner = inner.data

        if not isinstance(inner, dict):
            return None

        emoe = inner.get("emoe")
        if emoe is None:
            return None

        val = emoe.get("emoe_class_id")
        if val is None:
            return None

        if hasattr(val, "item"):
            val = val.item()

        return int(val)

    except Exception:
        return None


def load_traj_endpoint(traj_path: Path):
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))

        arr = np.array(raw["data"] if isinstance(raw, dict) else raw)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[-1, 0]), float(arr[-1, 1])

        return None

    except Exception:
        return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument("--labels_path", type=str, required=True)

    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="Class to inspect (0=left_turn, 2=right_turn)",
    )

    args = parser.parse_args()

    if args.class_id not in WRONG_SIDE:
        print(
            f"[WARN] class {args.class_id} has no wrong-side definition. "
            f"Defined for: {list(WRONG_SIDE.keys())}"
        )
        print("Showing all tokens for this class regardless of y sign.")

    cache_dir = Path(args.cache_dir)

    class_name = (
        EMOE_SCENE_TYPES[args.class_id]
        if args.class_id < len(EMOE_SCENE_TYPES)
        else f"class_{args.class_id}"
    )

    label_map = {}

    with open(args.labels_path) as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                r = json.loads(line)
                label_map[r["token"]] = r
            except Exception:
                continue

    print(f"[INFO] Loaded {len(label_map)} labels")

    print(
        f"[INFO] Looking for wrong-side class {args.class_id} "
        f"({class_name}) tokens\n"
    )

    wrong = []
    correct = 0
    skipped = 0

    for log_dir in sorted(cache_dir.iterdir()):

        if not log_dir.is_dir():
            continue

        for tag_dir in sorted(log_dir.iterdir()):

            if not tag_dir.is_dir():
                continue

            for tok_dir in sorted(tag_dir.iterdir()):

                if not tok_dir.is_dir():
                    continue

                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"

                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)

                if cid != args.class_id:
                    continue

                ep = load_traj_endpoint(traj_p)

                if ep is None:
                    skipped += 1
                    continue

                x, y = ep

                is_wrong = WRONG_SIDE.get(args.class_id, lambda _: False)(y)

                if is_wrong:

                    tok = tok_dir.name

                    rec = label_map.get(tok, {})

                    dbg = rec.get("debug", {})

                    wrong.append(
                        {
                            "token": tok,
                            "traj_x": x,
                            "traj_y": y,
                            "stage": rec.get("stage", "NOT_IN_LABELS"),
                            "scenario_type": rec.get("scenario_type", "?"),
                            "delta_h_deg": dbg.get("delta_heading_deg"),
                            "abs_dh_deg": dbg.get("abs_delta_heading_deg"),
                            "total_abs_deg": dbg.get("total_abs_heading_deg"),
                            "traversed_polygon": dbg.get(
                                "traversed_intersection_polygon"
                            ),
                            "traversed_connector": dbg.get(
                                "traversed_lane_connector"
                            ),
                            "conn_best_type": dbg.get("connector_best_type"),
                            "conn_best_ratio": dbg.get("connector_best_ratio"),
                            "lane_following_ok": dbg.get("lane_following_ok"),
                            "lane_med_err_deg": dbg.get(
                                "lane_following_median_err_deg"
                            ),
                            "gap_promoted": dbg.get(
                                "intersection_gap_promoted_to_straight"
                            ),
                            "fallback_promoted": dbg.get(
                                "intersection_fallback_promoted_to_straight"
                            ),
                            "tags": dbg.get("tags", []),
                        }
                    )

                else:

                    correct += 1

    print(f"  Correct (right side) : {correct}")
    print(f"  Wrong   (wrong side) : {len(wrong)}")
    print(f"  Skipped (load error) : {skipped}")

    if not wrong:
        print("\nNo misclassified tokens found.")
        return

    print("\n" + "─" * 90)
    print(f"  MISCLASSIFIED TOKENS — class {args.class_id} ({class_name})")
    print("─" * 90 + "\n")

    for i, w in enumerate(wrong):

        print(f"[{i+1}/{len(wrong)}] token={w['token'][:20]}")
        print(f"  traj endpoint        : x={w['traj_x']:+.2f}  y={w['traj_y']:+.2f}")
        print(f"  stage                : {w['stage']}")
        print(f"  scenario_type        : {w['scenario_type']}")
        print(f"  delta_h_deg          : {w['delta_h_deg']}")
        print(f"  abs_dh_deg           : {w['abs_dh_deg']}")
        print(f"  total_abs_deg        : {w['total_abs_deg']}")
        print(f"  traversed_polygon    : {w['traversed_polygon']}")
        print(f"  traversed_connector  : {w['traversed_connector']}")
        print(
            f"  conn_best_type       : {w['conn_best_type']}  "
            f"ratio={w['conn_best_ratio']}"
        )
        print(
            f"  lane_following_ok    : {w['lane_following_ok']}  "
            f"median_err={w['lane_med_err_deg']}"
        )
        print(f"  gap_promoted         : {w['gap_promoted']}")
        print(f"  fallback_promoted    : {w['fallback_promoted']}")
        print(f"  tags                 : {w['tags']}")
        print()

    stage_counts = Counter(w["stage"] for w in wrong)

    print("─" * 60)
    print("  Stage breakdown for wrong-side tokens:")

    for stage, cnt in stage_counts.most_common():
        print(f"  {cnt:>3}x  {stage}")

    print("─" * 60)


if __name__ == "__main__":
    main()
