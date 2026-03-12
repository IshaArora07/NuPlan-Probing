#!/usr/bin/env python3
"""
Diagnostic: check y-axis sign convention for left-turn trajectories.

Loads the first 10 class-0 (left_turn_at_intersection) tokens from
scene_labels.jsonl, finds their trajectory.gz in the cache, and prints
the endpoint (x, y) alongside the classifier’s delta_heading_deg.

Expected if conventions match:    delta_heading > 0  AND  endpoint_y > 0
Expected if y-axis is flipped:    delta_heading > 0  AND  endpoint_y < 0

Usage:
python check_y_convention.py \
--labels_path  ./emoe_precomputed/scene_labels.jsonl \
--cache_dir    ./nuplan_cache \
--class_id     0 \
--n_samples    10
"""

import gzip
import json
import pickle
import argparse
from pathlib import Path

import numpy as np


def build_token_index(cache_dir: Path) -> dict:
    """Walk cache_dir/log/tag/token/ and index token -> trajectory.gz path."""
    index = {}

    for log_dir in cache_dir.iterdir():
        if not log_dir.is_dir():
            continue

        for tag_dir in log_dir.iterdir():
            if not tag_dir.is_dir():
                continue

            for tok_dir in tag_dir.iterdir():
                if not tok_dir.is_dir():
                    continue

                p = tok_dir / "trajectory.gz"
                if p.exists():
                    index[tok_dir.name] = p

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="EMoE class to inspect (default 0 = left_turn)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of tokens to sample",
    )

    args = parser.parse_args()

    EMOE_SCENE_TYPES = [
        "left_turn_at_intersection",
        "straight_at_intersection",
        "right_turn_at_intersection",
        "straight_non_intersection",
        "roundabout",
        "u_turn",
        "others",
    ]

    class_name = (
        EMOE_SCENE_TYPES[args.class_id]
        if args.class_id < len(EMOE_SCENE_TYPES)
        else f"class_{args.class_id}"
    )

    print(f"\nInspecting class {args.class_id} ({class_name}), n={args.n_samples}\n")

    # Load matching tokens from labels
    tokens = []

    with open(args.labels_path) as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                r = json.loads(line)
            except Exception:
                continue

            if int(r.get("emoe_class_id", -1)) == args.class_id:
                tokens.append(
                    {
                        "token": r["token"],
                        "delta_h_deg": r.get("debug", {}).get("delta_heading_deg", None),
                        "endpoint_xy": r.get("endpoint_xy", None),
                        "travel_dist": r.get("travel_distance_m", None),
                    }
                )

            if len(tokens) >= args.n_samples:
                break

    print(f"Found {len(tokens)} class-{args.class_id} tokens in labels")

    # Build cache index
    print(f"Building cache index from {args.cache_dir} ...")

    index = build_token_index(Path(args.cache_dir))

    print(f"Indexed {len(index)} tokens with trajectory.gz\n")

    header = (
        f"{'token':<24s}  "
        f"{'delta_h (deg)':>14s}  "
        f"{'anch_x':>8s}  {'anch_y':>8s}  "
        f"{'traj_x':>8s}  {'traj_y':>8s}  "
        f"{'traj_h':>8s}  "
        f"{'y match?':>10s}"
    )

    print(header)
    print("-" * len(header))

    found = 0

    for t in tokens:

        tok = t["token"]

        if tok not in index:
            print(f"{tok[:24]:<24s}  {'— not in cache —':>14s}")
            continue

        try:

            raw = pickle.load(gzip.open(index[tok], "rb"))

            arr = np.array(raw["data"], dtype=np.float32)

            tx, ty, th = float(arr[-1, 0]), float(arr[-1, 1]), float(arr[-1, 2])

        except Exception as e:

            print(f"{tok[:24]:<24s}  load error: {e}")

            continue

        dh = t["delta_h_deg"]
        ep = t["endpoint_xy"]

        ax = float(ep[0]) if ep else float("nan")
        ay = float(ep[1]) if ep else float("nan")

        if dh is not None:

            classifier_left = dh > 0
            traj_left = ty > 0

            match = "✓ ok" if classifier_left == traj_left else "✗ FLIP"

        else:

            match = "?"

        print(
            f"{tok[:24]:<24s}  "
            f"{dh if dh is not None else float('nan'):>14.2f}  "
            f"{ax:>8.2f}  {ay:>8.2f}  "
            f"{tx:>8.2f}  {ty:>8.2f}  "
            f"{th:>8.4f}  "
            f"{match:>10s}"
        )

        found += 1

    print(f"\n{found} / {len(tokens)} tokens found in cache")

    print(
        """
Interpretation:
delta_h (deg)  : classifier’s net heading change — positive = left turn
anch_x / anch_y: endpoint from scene_labels.jsonl (used for KMeans anchors)
traj_x / traj_y: endpoint from trajectory.gz (what the model trains on)
y match?       : ✓ ok   → both agree on left/right
                 ✗ FLIP → y-axis sign is inverted between the two
"""
    )


if __name__ == "__main__":
    main()
