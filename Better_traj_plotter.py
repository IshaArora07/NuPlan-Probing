#!/usr/bin/env python3
"""
EMoE Trajectory Spaghetti Plotter

Reads class labels directly from features.gz (data["emoe"]["scene_label"])
so that class and trajectory always come from the same cache file.

Cache structure:
<cache_dir>/<log_name>/<scenario_tag>/<token>/features.gz
<cache_dir>/<log_name>/<scenario_tag>/<token>/trajectory.gz

trajectory.gz  →  {'data': ndarray (8, 3)}
8 waypoints at 1s intervals, columns (x, y, heading), ego-init frame

features.gz    →  {'data': PlutoFeature.data dict}
data["emoe"]["scene_label"]  →  int class id (if token was in labels)

Usage — explore:
python plot_trajectory_spaghetti.py \
--cache_dir ./nuplan_cache \
--mode explore

Usage — plot:
python plot_trajectory_spaghetti.py \
--cache_dir ./nuplan_cache \
--anchors_path ./emoe_precomputed/scene_anchors.npy \
--out_dir ./emoe_viz \
--mode plot \
--max_traj 300
"""

import gzip
import pickle
import argparse
import math
from pathlib import Path
from collections import defaultdict

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
                        "features":   feat_p if feat_p.exists() else None,
                        "log": log_dir.name,
                        "tag": tag_dir.name,
                    }
    return index

# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_gz_raw(path: Path):
    """Load gzip+pickle, unwrap {'data': …} shell, return inner object."""
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and list(obj.keys()) == ["data"]:
        return obj["data"]
    return obj

def load_trajectory(path: Path):
    """
    Returns (8, 2) float32 xy in ego-init frame, or None on failure.
    trajectory.gz: {'data': ndarray (8, 3)}  cols = (x, y, heading)
    """
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

from typing import Optional

def load_emoe_class(path: Path) -> Optional[int]:
    try:
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)

        # features.gz is saved as {"data": PlutoFeature_instance}
        # PlutoFeature_instance.data is the raw dict
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]

        # obj is now either a PlutoFeature instance or the raw data dict
        if hasattr(obj, "data"):          # PlutoFeature instance
            inner = obj.data
        elif isinstance(obj, dict):       # raw dict directly
            inner = obj
        else:
            return None

        emoe = inner.get("emoe")
        if emoe is None:
            return None

        val = emoe.get("emoe_class_id")   # correct key
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
    checked = 0

    for tok, entry in list(index.items())[:5]:
        print(f"\n  token : {tok}")
        print(f"  log   : {entry['log']}")
        print(f"  tag   : {entry['tag']}")

        # features.gz
        if entry["features"]:
            cid = load_emoe_class(entry["features"])
            if cid is not None:
                ok(f"  features.gz → emoe scene_label = {cid} ({EMOE_SCENE_TYPES[cid] if cid < 6 else 'others'})")
            else:
                warn("  features.gz → no emoe label (token not in scene_labels at cache time)")
        else:
            warn("  features.gz missing")

        # trajectory.gz
        if entry["trajectory"]:
            xy = load_trajectory(entry["trajectory"])
            if xy is not None:
                ok(f"  trajectory.gz → shape {xy.shape}  endpoint=({xy[-1,0]:.2f}, {xy[-1,1]:.2f}) m")
            else:
                err("  trajectory.gz → load_trajectory returned None")
        else:
            warn("  trajectory.gz missing")

        checked += 1

    hdr("Scanning all tokens for emoe labels…")
    for tok, entry in index.items():
        if entry["features"] is None:
            no_emoe += 1
            continue
        cid = load_emoe_class(entry["features"])
        if cid is None:
            no_emoe += 1
        else:
            class_counts[cid] += 1

    print(f"\n  Tokens with emoe label   : {sum(class_counts.values())}")
    print(f"  Tokens without emoe label: {no_emoe}")
    print(f"\n  Per-class counts:")
    for c in range(6):
        name = EMOE_SCENE_TYPES[c]
        n    = class_counts.get(c, 0)
        traj_ok = sum(
            1 for tok, e in index.items()
            if e["trajectory"] is not None
        )
        print(f"    class {c} ({name:<30s}): {n}")

    hdr("Summary")
    if sum(class_counts.values()) == 0:
        err("No emoe labels found in features.gz — was the feature cache built "
            "with the correct scene_labels.jsonl path in PlutoFeatureBuilder?")
    else:
        ok("emoe labels found → run --mode plot")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT MODE
# ──────────────────────────────────────────────────────────────────────────────

def plot_mode(cache_dir: Path, anchors_path: Path, out_dir: Path, max_traj: int):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    hdr("Building token index…")
    index = build_token_index(cache_dir)
    ok(f"Indexed {len(index)} tokens")

    # ── Read class label directly from features.gz ─────────────────────────
    hdr("Grouping tokens by class (from features.gz emoe label)…")
    by_class   = defaultdict(list)
    no_label   = 0
    no_traj    = 0

    for tok, entry in index.items():
        if entry["features"] is None:
            no_label += 1
            continue
        if entry["trajectory"] is None:
            no_traj += 1
            continue

        cid = load_emoe_class(entry["features"])
        if cid is None:
            no_label += 1
            continue
        if cid < 6:
            by_class[cid].append(tok)

    warn(f"{no_label} tokens had no emoe label in features.gz")
    if no_traj:
        warn(f"{no_traj} tokens had no trajectory.gz")

    print()
    for c in range(6):
        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:<30s}): {len(by_class[c])} tokens")

    if sum(len(v) for v in by_class.values()) == 0:
        err("No labelled tokens found. Was the cache built with scene_labels.jsonl "
            "path set correctly in PlutoFeatureBuilder._emoe_label_path?")
        return

    hdr("Loading anchors…")
    anchors = np.load(anchors_path)
    ok(f"Anchors shape: {anchors.shape}")

    hdr(f"Plotting (max {max_traj} per class)…")
    rng   = np.random.default_rng(seed=42)
    ncols = 3
    nrows = math.ceil(6 / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    parse_errors = 0

    for c in range(6):
        ax   = axes[c]
        col  = CLASS_COLORS[c]
        name = SHORT_NAMES[c]

        tokens = by_class[c]
        if len(tokens) > max_traj:
            idx    = rng.choice(len(tokens), size=max_traj, replace=False)
            tokens = [tokens[i] for i in idx]

        ok_count  = 0
        endpoints = []

        for tok in tokens:
            traj_path = index[tok]["trajectory"]
            xy = load_trajectory(traj_path)
            if xy is None or xy.shape[0] < 2:
                parse_errors += 1
                continue

            # Prepend ego origin so trajectory starts at (0,0)
            origin  = np.zeros((1, 2), dtype=np.float32)
            xy_full = np.concatenate([origin, xy], axis=0)

            ax.plot(xy_full[:, 0], xy_full[:, 1],
                    color=col, alpha=0.15, lw=0.9, zorder=2)
            ax.scatter(xy[-1, 0], xy[-1, 1],
                       s=12, color=col, alpha=0.5,
                       linewidths=0, zorder=3)
            endpoints.append(xy[-1])
            ok_count += 1

        # Anchors
        anc = anchors[c]
        ax.scatter(anc[:, 0], anc[:, 1],
                   s=200, color="#1A1A2E", marker="*",
                   zorder=6, linewidths=0.5, edgecolors="white",
                   label=f"Anchors (Ka={anc.shape[0]})")

        # Ego origin + forward arrow
        ax.scatter(0, 0, s=90, color="black", marker="D", zorder=7,
                   label="Ego origin")
        ax.annotate("", xy=(3, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        # Median endpoint distance in title
        if endpoints:
            ep_arr   = np.stack(endpoints)
            med_dist = float(np.median(np.linalg.norm(ep_arr, axis=1)))
            extra    = f"  |  median dist={med_dist:.1f} m"
        else:
            extra = ""

        ax.set_title(f"{name}\n(n={ok_count}{extra})",
                     fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("x  (ego-forward, m)", fontsize=10)
        ax.set_ylabel("y  (ego-left, m)",    fontsize=10)
        ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:<30s}): {ok_count} plotted")

    for i in range(6, len(axes)):
        axes[i].set_visible(False)

    if parse_errors:
        warn(f"{parse_errors} trajectory files failed to parse")

    fig.suptitle(
        "EMoE Planner — GT Trajectory Spaghetti + KMeans Anchors  (ego frame)\n"
        "Class label read from features.gz  |  line = 8s GT future  "
        "|  dot = endpoint  |  ★ = anchor",
        fontsize=13, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig4_trajectory_spaghetti.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    ok(f"Saved → {out_path}")

# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",    type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir",      type=str, default="./emoe_viz")
    parser.add_argument("--mode",         type=str, default="explore",
                        choices=["explore", "plot"])
    parser.add_argument("--max_traj",     type=int, default=300)
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
            args.max_traj,
        )

if __name__ == "__main__":
    main()
