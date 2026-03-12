#!/usr/bin/env python3
"""
Deep check for misclassified tokens.

For each wrong-side token, loads:

1. trajectory.gz — what direction did ego actually go
2. scene_labels.jsonl — what delta_heading did the precompute script see

Then re-derives the endpoint direction from delta_heading to see
if the discrepancy is in the precompute or in the trajectory.

Usage:
python deep_check_misclassified.py \
--cache_dir   ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--class_id    0
"""

import gzip
import json
import math
import pickle
import argparse
from pathlib import Path

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

WRONG_SIDE = {
    0: lambda y: y < 0,
    2: lambda y: y > 0,
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


def load_full_trajectory(traj_path: Path):
    """Load full trajectory array from trajectory.gz."""
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))
        arr = np.array(raw["data"] if isinstance(raw, dict) else raw)
        return arr
    except Exception:
        return None


def predicted_endpoint_from_delta_h(delta_h_deg: float, dist: float = 30.0):
    """
    Given a net heading change (degrees) and approximate travel distance,
    estimate where ego ends up in ego frame (x=forward, y=left).
    """
    dh = math.radians(delta_h_deg)
    avg = dh / 2.0
    dx = dist * math.cos(avg)
    dy = dist * math.sin(avg)
    return dx, dy


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--class_id", type=int, default=0)

    args = parser.parse_args()

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
    print(f"[INFO] Deep-checking wrong-side class {args.class_id} ({class_name})\n")

    wrong_side_fn = WRONG_SIDE.get(args.class_id, lambda y: False)

    found = 0

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

                arr = load_full_trajectory(traj_p)

                if arr is None:
                    continue

                last_y = float(arr[-1, 1]) if arr.shape[1] >= 2 else 0.0
                last_x = float(arr[-1, 0])

                if not wrong_side_fn(last_y):
                    continue

                found += 1
                tok = tok_dir.name

                rec = label_map.get(tok, {})
                dbg = rec.get("debug", {})

                delta_h = dbg.get("delta_heading_deg")
                ep_xy = rec.get("endpoint_xy")
                dist = rec.get("travel_distance_m", 30.0)

                print("=" * 70)
                print(f"[{found}] token = {tok}")
                print("=" * 70)

                print("\n  ── trajectory.gz ──")
                print(f"  shape            : {arr.shape}")
                print(
                    f"  cols             : {'(x,y,heading)' if arr.shape[1]==3 else '(x,y)'}"
                )

                for t in range(arr.shape[0]):
                    row = "  ".join(f"{v:+8.3f}" for v in arr[t])
                    marker = (
                        " ← present+1s"
                        if t == 0
                        else (" ← ENDPOINT" if t == arr.shape[0] - 1 else "")
                    )
                    print(f"  t={t+1}s  [{row}]{marker}")

                traj_dir = "LEFT (y>0)" if last_y > 0 else "RIGHT (y<0)"

                print(
                    f"\n  trajectory endpoint : x={last_x:+.2f}  y={last_y:+.2f}  → {traj_dir}"
                )

                print("\n  ── precompute (scene_labels.jsonl) ──")
                print(f"  stage            : {rec.get('stage', '?')}")
                print(f"  delta_h_deg      : {delta_h}")
                print(f"  abs_dh_deg       : {dbg.get('abs_delta_heading_deg')}")
                print(f"  travel_dist_m    : {dist:.2f}")
                print(f"  endpoint_xy      : {ep_xy}")

                if ep_xy:
                    ep_dir = "LEFT (y>0)" if ep_xy[1] > 0 else "RIGHT (y<0)"
                    print(f"  endpoint direction: {ep_dir}")

                print("\n  ── Agreement check ──")

                if ep_xy:

                    precompute_left = ep_xy[1] > 0
                    traj_left = last_y > 0

                    if precompute_left == traj_left:
                        print("  ✓ precompute endpoint and trajectory AGREE on direction")
                        print(f"    → both say {'LEFT' if traj_left else 'RIGHT'}")
                        print("    → classifier assigned wrong class despite correct geometry")
                        print("    → BUG IS IN CLASSIFIER LOGIC")
                    else:
                        print("  ✗ precompute endpoint and trajectory DISAGREE")
                        print(
                            f"    precompute says: {'LEFT' if precompute_left else 'RIGHT'}"
                        )
                        print(
                            f"    trajectory says: {'LEFT' if traj_left else 'RIGHT'}"
                        )
                        print("    → FRAME MISMATCH between precompute and trajectory.gz")

                else:
                    print("  ? endpoint_xy not in labels — cannot compare")

                print("\n  ── connector ──")
                print(f"  conn_best_type   : {dbg.get('connector_best_type')}")
                print(f"  conn_best_ratio  : {dbg.get('connector_best_ratio')}")
                print(f"  turn_counts      : {dbg.get('connector_turn_counts')}")
                print(f"  lane_follow_ok   : {dbg.get('lane_following_ok')}")
                print(f"  lane_med_err_deg : {dbg.get('lane_following_median_err_deg')}")
                print()

    if found == 0:
        print("No wrong-side tokens found.")
    else:
        print(f"\n[DONE] Found {found} wrong-side tokens for class {args.class_id}")
        print(
            """
Summary of what to look for:
✓ precompute + traj AGREE  → classifier logic bug
✗ precompute + traj DISAGREE → frame mismatch
? no endpoint_xy in labels → rebuild labels with endpoint_xy
"""
        )


if __name__ == "__main__":
    main()
