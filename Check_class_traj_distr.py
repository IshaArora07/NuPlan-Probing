#!/usr/bin/env python3
"""
Check trajectory endpoint y-distribution for each class,
reading class label directly from features.gz.

Usage:
python check_class_traj_distribution.py \
--cache_dir ./nuplan_cache \
--class_id 0
"""

import gzip
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

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


def load_emoe_class(feat_path: Path):
    """Read emoe_class_id from features.gz. Returns None if not present."""
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
    """Load last (x, y) from trajectory.gz. Returns None on failure."""
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))

        if isinstance(raw, dict):
            arr = np.array(raw["data"])
        else:
            arr = np.array(raw)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[-1, 0]), float(arr[-1, 1])

        return None

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Class to focus on (default: all classes)",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    y_vals = defaultdict(list)
    x_vals = defaultdict(list)

    no_label = 0
    no_traj = 0
    errors = 0

    print(f"[INFO] Scanning {cache_dir} ...")

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

                if not feat_p.exists():
                    no_label += 1
                    continue

                if not traj_p.exists():
                    no_traj += 1
                    continue

                cid = load_emoe_class(feat_p)

                if cid is None:
                    no_label += 1
                    continue

                ep = load_traj_endpoint(traj_p)

                if ep is None:
                    errors += 1
                    continue

                x, y = ep

                y_vals[cid].append(y)
                x_vals[cid].append(x)

    print(f"\n[INFO] no emoe label : {no_label}")
    print(f"[INFO] no trajectory : {no_traj}")
    print(f"[INFO] load errors   : {errors}")

    classes = [args.class_id] if args.class_id is not None else list(range(7))

    print("\n" + "─" * 75)
    print(
        f"  {'class':<4}  {'name':<30}  {'n':>5}  "
        f"{'y>0':>6}  {'y<0':>6}  {'%y<0':>7}  "
        f"{'med_y':>8}  {'med_x':>8}"
    )
    print("─" * 75)

    for c in classes:

        ys = np.array(y_vals[c]) if y_vals[c] else np.array([])
        xs = np.array(x_vals[c]) if x_vals[c] else np.array([])

        n = len(ys)

        name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else f"class_{c}"

        if n == 0:
            print(f"  {c:<4}  {name:<30}  {'0':>5}  —")
            continue

        above = int((ys > 0).sum())
        below = int((ys < 0).sum())
        pct = 100.0 * below / n

        print(
            f"  {c:<4}  {name:<30}  {n:>5}  "
            f"{above:>6}  {below:>6}  {pct:>6.1f}%  "
            f"{np.median(ys):>8.2f}  {np.median(xs):>8.2f}"
        )

    print("─" * 75)

    if args.class_id is not None:

        c = args.class_id

        ys = np.array(y_vals[c]) if y_vals[c] else np.array([])
        xs = np.array(x_vals[c]) if x_vals[c] else np.array([])

        if len(ys) == 0:
            print(f"\nNo labelled tokens found for class {c}")
            return

        name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else "?"

        print(f"\n[DETAIL] class {c} ({name})")

        print(
            f"  y percentiles: "
            f"p5={np.percentile(ys,5):.2f}  "
            f"p25={np.percentile(ys,25):.2f}  "
            f"p50={np.percentile(ys,50):.2f}  "
            f"p75={np.percentile(ys,75):.2f}  "
            f"p95={np.percentile(ys,95):.2f}"
        )

        print(
            f"  x percentiles: "
            f"p5={np.percentile(xs,5):.2f}  "
            f"p25={np.percentile(xs,25):.2f}  "
            f"p50={np.percentile(xs,50):.2f}  "
            f"p75={np.percentile(xs,75):.2f}  "
            f"p95={np.percentile(xs,95):.2f}"
        )

        print(f"  y < 0  : {int((ys<0).sum())} / {len(ys)}")
        print(f"  y = 0  : {int((ys==0).sum())} / {len(ys)}")
        print(f"  y > 0  : {int((ys>0).sum())} / {len(ys)}")

        worst_idx = np.argsort(ys)[:5]

        print("\n  5 most negative y endpoints (likely misclassified):")
        print(f"  {'y':>8}  {'x':>8}")

        for i in worst_idx:
            print(f"  {ys[i]:>8.2f}  {xs[i]:>8.2f}")


if __name__ == "__main__":
    main()
