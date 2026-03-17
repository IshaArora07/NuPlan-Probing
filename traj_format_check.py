#!/usr/bin/env python3
"""
Check trajectory.gz format — absolute positions vs deltas, scale, waypoint intervals.
"""

import gzip
import json
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


def load_trajectory(traj_path: Path):
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))
        arr = np.array(raw["data"] if isinstance(raw, dict) else raw)
        return arr
    except Exception:
        return None


def analyse_trajectory(arr: np.ndarray, tok: str, label_ep=None):

    print("\n" + "=" * 70)
    print(f"  token : {tok[:24]}")
    print(
        f"  shape : {arr.shape}  "
        f"cols={'(x,y,h)' if arr.shape[1]==3 else '(x,y)'}"
    )
    print("=" * 70)

    n = arr.shape[0]
    has_heading = arr.shape[1] >= 3

    print(f"\n  All waypoints (assumed 1s intervals → {n}s total):")
    print(
        f"  {'t':>4s}  {'x':>10s}  {'y':>10s}"
        + (f"  {'heading':>10s}" if has_heading else "")
    )

    print(
        f"  {'─'*4}  {'─'*10}  {'─'*10}"
        + (f"  {'─'*10}" if has_heading else "")
    )

    for i in range(n):
        h_str = f"  {arr[i,2]:>10.4f}" if has_heading else ""
        print(f"  {i+1:>4d}  {arr[i,0]:>10.3f}  {arr[i,1]:>10.3f}{h_str}")

    diffs = np.diff(arr[:, :2], axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)

    print(f"\n  Step-to-step deltas:")
    print(f"  {'step':>6s}  {'dx':>10s}  {'dy':>10s}  {'dist':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")

    for i, (d, dist) in enumerate(zip(diffs, step_dists)):
        print(f"  {i}→{i+1:>3d}  {d[0]:>10.3f}  {d[1]:>10.3f}  {dist:>8.3f}m")

    total_path = float(step_dists.sum())
    straight_dist = float(np.linalg.norm(arr[-1, :2] - arr[0, :2])) if n > 1 else 0.0

    print(f"\n  Total path length  : {total_path:.2f} m")
    print(f"  Straight-line dist : {straight_dist:.2f} m")
    print(f"  Final endpoint     : x={arr[-1,0]:+.3f}  y={arr[-1,1]:+.3f}")

    print(f"\n  ── Scale diagnosis ──")
    avg_speed = total_path / n
    print(f"  Avg speed (if 1s intervals) : {avg_speed:.2f} m/s")

    if avg_speed > 50:
        print("  ✗ Unrealistic high speed → wrong timestep or units")
    elif avg_speed < 0.1:
        print("  ✗ Unrealistic low speed → normalized or scaled data")
    else:
        print("  ✓ Speed plausible")

    print(f"\n  ── Absolute vs delta ──")
    if np.allclose(arr[0, :2], [0.0, 0.0], atol=0.5):
        print("  first waypoint ≈ (0,0) → likely RELATIVE")
    else:
        print(
            f"  first waypoint ({arr[0,0]:.2f}, {arr[0,1]:.2f}) ≠ (0,0)"
        )
        print("  → likely ABSOLUTE ego-frame positions")

    if label_ep is not None:

        print(f"\n  ── Anchor endpoint comparison ──")
        print(
            f"  anchor endpoint_xy : x={label_ep[0]:+.3f}  y={label_ep[1]:+.3f}"
        )
        print(
            f"  traj   endpoint    : x={arr[-1,0]:+.3f}  y={arr[-1,1]:+.3f}"
        )

        scale_x = arr[-1, 0] / label_ep[0] if abs(label_ep[0]) > 0.1 else float("nan")
        scale_y = arr[-1, 1] / label_ep[1] if abs(label_ep[1]) > 0.1 else float("nan")

        print(f"  scale ratio x      : {scale_x:.3f}")
        print(f"  scale ratio y      : {scale_y:.3f}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--class_id", type=int, default=1)
    parser.add_argument("--token", type=str, default=None)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    label_map = {}

    if args.labels_path:
        with open(args.labels_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    label_map[r["token"]] = r
                except Exception:
                    continue

    if args.token:
        matches = list(cache_dir.glob(f"*/*/{args.token}"))
        for tok_dir in matches:
            traj_p = tok_dir / "trajectory.gz"
            if traj_p.exists():
                arr = load_trajectory(traj_p)
                if arr is not None:
                    rec = label_map.get(args.token, {})
                    analyse_trajectory(arr, args.token, rec.get("endpoint_xy"))
        return

    found = 0

    for log_dir in cache_dir.iterdir():
        if not log_dir.is_dir():
            continue

        for tag_dir in log_dir.iterdir():
            if not tag_dir.is_dir():
                continue

            for tok_dir in tag_dir.iterdir():
                if not tok_dir.is_dir():
                    continue

                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"

                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)
                if cid != args.class_id:
                    continue

                arr = load_trajectory(traj_p)
                if arr is None:
                    continue

                tok = tok_dir.name
                rec = label_map.get(tok, {})

                analyse_trajectory(arr, tok, rec.get("endpoint_xy"))

                found += 1
                if found >= args.n_samples:
                    return

    if found == 0:
        print(f"No class {args.class_id} tokens found.")


if __name__ == "__main__":
    main()
