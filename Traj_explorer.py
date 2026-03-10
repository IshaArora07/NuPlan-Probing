#!/usr/bin/env python3
"""
EMoE Cache Explorer + Trajectory Loader
"""

import gzip
import json
import pickle
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── ANSI ─────────────────────────────────────────────────────

GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
BLU = "\033[94m"
RST = "\033[0m"
BOLD = "\033[1m"


def ok(m):
    print(f"{GRN} ✓ {m}{RST}")


def warn(m):
    print(f"{YLW} ⚠ {m}{RST}")


def err(m):
    print(f"{RED} ✗ {m}{RST}")


def hdr(m):
    print(f"\n{BOLD}{BLU}{'─'*60}\n  {m}\n{'─'*60}{RST}")


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
]

SHORT_NAMES = [
    "Left turn\n(intersection)",
    "Straight\n(intersection)",
    "Right turn\n(intersection)",
    "Straight\n(non-intersection)",
    "Roundabout",
    "U-turn",
]

CLASS_COLORS = [
    "#E63946",
    "#2196F3",
    "#FF9800",
    "#4CAF50",
    "#9C27B0",
    "#00BCD4",
]


# ─────────────────────────────────────────────
# Build token index
# ─────────────────────────────────────────────

def build_token_index(cache_dir: Path) -> dict:
    """
    Returns dict: token -> paths
    """

    index = {}
    cache_dir = cache_dir.resolve()

    log_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    print(f"Found {len(log_dirs)} log directories")

    for log_dir in log_dirs:
        for tag_dir in log_dir.iterdir():
            if not tag_dir.is_dir():
                continue

            for tok_dir in tag_dir.iterdir():
                if not tok_dir.is_dir():
                    continue

                token = tok_dir.name

                traj_p = tok_dir / "trajectory.gz"
                feat_p = tok_dir / "features.gz"

                if traj_p.exists():
                    index[token] = {
                        "trajectory": traj_p,
                        "features": feat_p if feat_p.exists() else None,
                        "log": log_dir.name,
                        "tag": tag_dir.name,
                    }

    return index


# ─────────────────────────────────────────────
# Load gzip pickle
# ─────────────────────────────────────────────

def load_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Pretty print structure
# ─────────────────────────────────────────────

def _print_nested(data, indent=0, max_depth=5, depth=0, label=""):
    pad = " " * indent
    prefix = f"{pad}{label}: " if label else pad

    if depth > max_depth:
        print(prefix + "...")
        return

    if isinstance(data, dict):
        print(prefix + f"dict ({len(data)} keys)")
        for k in list(data.keys())[:12]:
            _print_nested(data[k], indent + 4, max_depth, depth + 1, label=repr(k))

    elif isinstance(data, (list, tuple)):
        print(prefix + f"{type(data).__name__} len={len(data)}")
        if data:
            _print_nested(data[0], indent + 4, max_depth, depth + 1, label="[0]")

    elif hasattr(data, "shape") and hasattr(data, "dtype"):
        print(prefix + f"array/tensor shape={tuple(data.shape)} dtype={data.dtype}")

    else:
        s = str(data)
        print(prefix + s[:100] + ("..." if len(s) > 100 else ""))


# ─────────────────────────────────────────────
# Extract ego trajectory
# ─────────────────────────────────────────────

def extract_ego_trajectory(traj_data):

    if isinstance(traj_data, np.ndarray):
        if traj_data.ndim == 2 and traj_data.shape[-1] >= 2:
            return traj_data.astype(np.float32)

    try:
        import torch
        if isinstance(traj_data, torch.Tensor):
            arr = traj_data.cpu().numpy()
            if arr.ndim == 2:
                return arr.astype(np.float32)
            if arr.ndim == 3:
                return arr[0].astype(np.float32)
    except ImportError:
        pass

    if isinstance(traj_data, dict):
        for k in [
            "ego_trajectory",
            "trajectory",
            "ego",
            "gt_trajectory",
            "future_trajectory",
            "poses",
        ]:
            if k in traj_data:
                return extract_ego_trajectory(traj_data[k])

    if isinstance(traj_data, (list, tuple)) and len(traj_data) > 0:
        first = traj_data[0]

        if hasattr(first, "x") and hasattr(first, "y"):
            return np.array([[s.x, s.y] for s in traj_data], dtype=np.float32)

        if hasattr(first, "rear_axle"):
            return np.array(
                [[s.rear_axle.x, s.rear_axle.y] for s in traj_data],
                dtype=np.float32,
            )

        try:
            arr = np.array(traj_data, dtype=np.float32)
            if arr.ndim == 2:
                return arr
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────
# Ego frame conversion
# ─────────────────────────────────────────────

def to_ego_frame(xy: np.ndarray):

    x0, y0 = float(xy[0, 0]), float(xy[0, 1])

    dx = float(xy[1, 0] - x0) if len(xy) > 1 else 1.0
    dy = float(xy[1, 1] - y0) if len(xy) > 1 else 0.0

    theta = math.atan2(dy, dx)

    c, s = math.cos(-theta), math.sin(-theta)

    rel = xy - np.array([x0, y0], dtype=np.float32)

    rot = np.stack(
        [
            c * rel[:, 0] - s * rel[:, 1],
            s * rel[:, 0] + c * rel[:, 1],
        ],
        axis=1,
    )

    return rot.astype(np.float32)


# ─────────────────────────────────────────────
# Explore mode
# ─────────────────────────────────────────────

def explore_mode(cache_dir: Path, labels_path: Path):

    hdr("Building token index")
    index = build_token_index(cache_dir)
    ok(f"Indexed {len(index)} tokens")

    records = []

    with labels_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    found = sum(1 for r in records if r["token"] in index)

    warn(f"{found}/{len(records)} label tokens found in cache")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./emoe_viz")

    parser.add_argument(
        "--mode",
        type=str,
        default="explore",
        choices=["explore", "plot"],
    )

    parser.add_argument(
        "--max_traj",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if args.mode == "explore":
        explore_mode(Path(args.cache_dir), Path(args.labels_path))
    else:
        if args.anchors_path is None:
            raise ValueError("--anchors_path required for plot mode")


if __name__ == "__main__":
    main()
