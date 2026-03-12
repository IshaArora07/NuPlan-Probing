#!/usr/bin/env python3

import gzip
import pickle
import argparse
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np


# ── ANSI ──────────────────────────────────────────────────────────────────────

GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
BLU = "\033[94m"; RST = "\033[0m";  BOLD = "\033[1m"

def ok(m):   print(f"{GRN}  ✓ {m}{RST}")
def warn(m): print(f"{YLW}  ⚠ {m}{RST}")
def err(m):  print(f"{RED}  ✗ {m}{RST}")
def hdr(m):  print(f"\n{BOLD}{BLU}{'─'*60}\n  {m}\n{'─'*60}{RST}")


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

CLASS_COLORS = ["#E63946", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"]


# ──────────────────────────────────────────────────────────────────────────────
# Cache indexing
# ──────────────────────────────────────────────────────────────────────────────

def build_token_index(cache_dir: Path) -> dict:
    """token -> {"trajectory": Path|None, "features": Path|None, "log", "tag"}"""

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

                traj_p = tok_dir / "trajectory.gz"
                feat_p = tok_dir / "features.gz"

                if traj_p.exists() or feat_p.exists():
                    index[tok_dir.name] = {
                        "trajectory": traj_p if traj_p.exists() else None,
                        "features": feat_p if feat_p.exists() else None,
                        "log": log_dir.name,
                        "tag": tag_dir.name,
                    }

    return index


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_gz_raw(path: Path):
    """Load gzip+pickle and unwrap {'data': ...} shell."""
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and list(obj.keys()) == ["data"]:
        return obj["data"]

    return obj


def load_trajectory(path: Path) -> Optional[np.ndarray]:
    try:
        arr = load_gz_raw(path)

        if hasattr(arr, "numpy"):
            arr = arr.numpy()

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]

        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] >= 2:
            return arr[0, :, :2]

        return None

    except Exception:
        return None


def load_emoe_class(path: Path) -> Optional[int]:

    try:

        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]

        if hasattr(obj, "data"):
            inner = obj.data
        elif isinstance(obj, dict):
            inner = obj
        else:
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


# ──────────────────────────────────────────────────────────────────────────────
# EXPLORE MODE
# ──────────────────────────────────────────────────────────────────────────────

def explore_mode(cache_dir: Path):

    hdr("Building token index…")
    index = build_token_index(cache_dir)

    ok(f"Indexed {len(index)} tokens")

    hdr("Sampling 5 tokens — checking features.gz + trajectory.gz…")

    class_counts = defaultdict(int)
    no_emoe = 0

    for tok, entry in list(index.items())[:5]:

        print(f"\n  token : {tok}")
        print(f"  log   : {entry['log']}")
        print(f"  tag   : {entry['tag']}")

        if entry["features"]:
            cid = load_emoe_class(entry["features"])
            if cid is not None:
                ok(f"  features.gz → emoe scene_label = {cid}")
            else:
                warn("  features.gz → no emoe label")
        else:
            warn("  features.gz missing")

        if entry["trajectory"]:
            xy = load_trajectory(entry["trajectory"])
            if xy is not None:
                ok(f"  trajectory.gz → shape {xy.shape}")
            else:
                err("  trajectory.gz → load_trajectory returned None")
        else:
            warn("  trajectory.gz missing")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT MODE
# ──────────────────────────────────────────────────────────────────────────────

def plot_mode(cache_dir: Path, anchors_path: Path, out_dir: Path, max_traj: int):

    import matplotlib.pyplot as plt

    hdr("Building token index…")
    index = build_token_index(cache_dir)

    ok(f"Indexed {len(index)} tokens")

    by_class = defaultdict(list)

    for tok, entry in index.items():

        if entry["features"] is None:
            continue

        if entry["trajectory"] is None:
            continue

        cid = load_emoe_class(entry["features"])

        if cid is None:
            continue

        if cid < 6:
            by_class[cid].append(tok)

    anchors = np.load(anchors_path)

    rng = np.random.default_rng(seed=42)

    ncols = 3
    nrows = math.ceil(6 / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7*ncols,7*nrows),
                             constrained_layout=True)

    axes = np.array(axes).flatten()

    for c in range(6):

        ax = axes[c]

        tokens = by_class[c]

        if len(tokens) > max_traj:
            idx = rng.choice(len(tokens), size=max_traj, replace=False)
            tokens = [tokens[i] for i in idx]

        for tok in tokens:

            traj_path = index[tok]["trajectory"]

            xy = load_trajectory(traj_path)

            if xy is None:
                continue

            origin = np.zeros((1,2),dtype=np.float32)

            xy_full = np.concatenate([origin,xy],axis=0)

            ax.plot(xy_full[:,0],xy_full[:,1],alpha=0.15)

        anc = anchors[c]

        ax.scatter(anc[:,0],anc[:,1],marker="*",s=200)

        ax.scatter(0,0,marker="D")

        ax.set_title(SHORT_NAMES[c])

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fig4_trajectory_spaghetti.png"

    fig.savefig(out_path, dpi=150)

    plt.close(fig)

    ok(f"Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./emoe_viz")
    parser.add_argument("--mode", type=str, default="explore",
                        choices=["explore","plot"])
    parser.add_argument("--max_traj", type=int, default=300)

    args = parser.parse_args()

    if args.mode == "explore":
        explore_mode(Path(args.cache_dir))

    else:

        if args.anchors_path is None:
            raise ValueError("--anchors_path required for plot mode")

        plot_mode(
            Path(args.cache_dir),
            Path(args.anchors_path),
            Path(args.out_dir),
            args.max_traj
        )


if __name__ == "__main__":
    main()
