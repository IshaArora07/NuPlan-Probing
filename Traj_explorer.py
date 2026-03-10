#!/usr/bin/env python3
"""
EMoE Cache Explorer + Trajectory Loader

Understands PLUTO cache structure:
<cache_dir>/<log_name>/<scenario_tag>/<token>/features.gz
<cache_dir>/<log_name>/<scenario_tag>/<token>/trajectory.gz

Usage — just explore structure:
python explore_cache_trajectories.py \
--cache_dir ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--mode explore

Usage — load trajectories and produce spaghetti plots:
python explore_cache_trajectories.py \
--cache_dir ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--anchors_path ./emoe_precomputed/scene_anchors.npy \
--mode plot \
--out_dir ./emoe_viz \
--max_traj 300
"""

import gzip
import json
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
# 1.  Build token → path index
# Walks: cache_dir / log / tag / token / {features,trajectory}.gz
# ──────────────────────────────────────────────────────────────────────────────

def build_token_index(cache_dir: Path) -> dict:
    """
    Returns dict: token_str -> {"features": Path, "trajectory": Path, "log": str, "tag": str}
    Only includes tokens that have at least a trajectory.gz.
    """
    index = {}
    cache_dir = cache_dir.resolve()

    log_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    print(f"  Found {len(log_dirs)} log directories")

    for log_dir in log_dirs:
        tag_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
        for tag_dir in tag_dirs:
            token_dirs = [d for d in tag_dir.iterdir() if d.is_dir()]
            for tok_dir in token_dirs:
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

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Load a single .gz file (gzip + pickle)
# ──────────────────────────────────────────────────────────────────────────────

def load_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Pretty-print nested structure (for exploration)
# ──────────────────────────────────────────────────────────────────────────────

def _print_nested(data, indent=0, max_depth=5, depth=0, label=""):
    pad = " " * indent
    prefix = f"{pad}{label}: " if label else pad

    if depth > max_depth:
        print(prefix + "...")
        return

    if isinstance(data, dict):
        print(prefix + f"dict  ({len(data)} keys)")
        for k in list(data.keys())[:12]:
            _print_nested(data[k], indent + 4, max_depth, depth + 1, label=repr(k))
    elif isinstance(data, (list, tuple)):
        print(prefix + f"{type(data).__name__}  len={len(data)}")
        if data:
            _print_nested(data[0], indent + 4, max_depth, depth + 1, label="[0]")
    elif hasattr(data, "shape") and hasattr(data, "dtype"):
        print(prefix + f"array/tensor  shape={tuple(data.shape)}  dtype={data.dtype}")
    else:
        s = str(data)
        print(prefix + s[:100] + ("…" if len(s) > 100 else ""))

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Extract ego trajectory array from trajectory.gz
# Handles several common PLUTO formats
# ──────────────────────────────────────────────────────────────────────────────

def extract_ego_trajectory(traj_data) -> np.ndarray | None:
    """
    Tries to return ego trajectory as np.ndarray shape (T, 2) or (T, 3)
    in whatever coordinate frame is stored (usually global or ego-init frame).

    Returns None if structure is unrecognised — caller should inspect manually.
    """
    # ── Case A: raw ndarray ────────────────────────────────────────────────────
    if isinstance(traj_data, np.ndarray):
        if traj_data.ndim == 2 and traj_data.shape[-1] >= 2:
            return traj_data.astype(np.float32)

    # ── Case B: torch tensor ──────────────────────────────────────────────────
    try:
        import torch
        if isinstance(traj_data, torch.Tensor):
            arr = traj_data.cpu().numpy()
            if arr.ndim == 2 and arr.shape[-1] >= 2:
                return arr.astype(np.float32)
            if arr.ndim == 3:          # (1, T, D) batch dim
                return arr[0].astype(np.float32)
    except ImportError:
        pass

    # ── Case C: dict with common keys ─────────────────────────────────────────
    if isinstance(traj_data, dict):
        for k in ["ego_trajectory", "trajectory", "ego", "gt_trajectory",
                  "future_trajectory", "poses"]:
            if k in traj_data:
                return extract_ego_trajectory(traj_data[k])

    # ── Case D: object with .data or .array attribute ─────────────────────────
    for attr in ["data", "array", "numpy", "poses"]:
        if hasattr(traj_data, attr):
            v = getattr(traj_data, attr)
            if callable(v):
                v = v()
            result = extract_ego_trajectory(v)
            if result is not None:
                return result

    # ── Case E: list of states/poses ──────────────────────────────────────────
    if isinstance(traj_data, (list, tuple)) and len(traj_data) > 0:
        first = traj_data[0]
        # list of objects with .x .y attributes
        if hasattr(first, "x") and hasattr(first, "y"):
            xy = np.array([[s.x, s.y] for s in traj_data], dtype=np.float32)
            return xy
        # list of objects with .rear_axle
        if hasattr(first, "rear_axle"):
            xy = np.array([[s.rear_axle.x, s.rear_axle.y] for s in traj_data],
                          dtype=np.float32)
            return xy
        # list of 2- or 3-tuples
        try:
            arr = np.array(traj_data, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] >= 2:
                return arr
        except Exception:
            pass

    return None   # unrecognised — caller will print structure

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Convert global-frame trajectory to ego-init frame
# ──────────────────────────────────────────────────────────────────────────────

def to_ego_frame(xy: np.ndarray) -> np.ndarray:
    """
    Rotate + translate so that xy[0] is origin and heading(xy[0]→xy[1]) is x-axis.
    Returns (T, 2) in ego frame.
    """
    x0, y0 = float(xy[0, 0]), float(xy[0, 1])
    dx = float(xy[1, 0] - x0) if len(xy) > 1 else 1.0
    dy = float(xy[1, 1] - y0) if len(xy) > 1 else 0.0
    theta = math.atan2(dy, dx)
    c, s = math.cos(-theta), math.sin(-theta)

    rel = xy - np.array([x0, y0], dtype=np.float32)
    rot = np.stack([c * rel[:, 0] - s * rel[:, 1],
                    s * rel[:, 0] + c * rel[:, 1]], axis=1)
    return rot.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Explore mode: print structure of a few files
# ──────────────────────────────────────────────────────────────────────────────

def explore_mode(cache_dir: Path, labels_path: Path):
    hdr("Building token index…")
    index = build_token_index(cache_dir)
    ok(f"Indexed {len(index)} tokens with trajectory.gz")

    hdr("Checking label tokens against index")
    records = []
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    found = sum(1 for r in records if r["token"] in index)
    warn(f"{found}/{len(records)} label tokens found in cache index")

    hdr("Sample trajectory.gz structure (first 3 found tokens)")
    checked = 0
    for rec in records:
        tok = rec["token"]
        if tok not in index:
            continue
        entry = index[tok]
        print(f"\n  token : {tok}")
        print(f"  log   : {entry['log']}")
        print(f"  tag   : {entry['tag']}")
        print(f"  traj  : {entry['trajectory']}")

        try:
            traj_data = load_gz(entry["trajectory"])
            print("  trajectory.gz contents:")
            _print_nested(traj_data, indent=4)

            xy = extract_ego_trajectory(traj_data)
            if xy is not None:
                ok(f"  extract_ego_trajectory → shape {xy.shape}")
            else:
                warn("  extract_ego_trajectory returned None — needs manual inspection above")
        except Exception as e:
            err(f"  Failed to load trajectory.gz: {e}")

        if entry["features"] is not None:
            try:
                feat_data = load_gz(entry["features"])
                print("  features.gz contents:")
                _print_nested(feat_data, indent=4, max_depth=3)
            except Exception as e:
                err(f"  Failed to load features.gz: {e}")

        checked += 1
        if checked >= 3:
            break

    hdr("Summary")
    print(f"  Total cached tokens : {len(index)}")
    print(f"  Label tokens found  : {found} / {len(records)}")
    print("""
Next step:
If extract_ego_trajectory printed a valid shape above → run with --mode plot
If it printed None → share the trajectory.gz structure printout above
and we’ll fix the extractor for your specific format.
""")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Plot mode: spaghetti plots per class
# ──────────────────────────────────────────────────────────────────────────────

def plot_mode(cache_dir: Path, labels_path: Path, anchors_path: Path,
              out_dir: Path, max_traj: int):
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

    hdr("Loading labels…")
    records = []
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    # Group tokens by class, keep only those in index
    by_class = defaultdict(list)
    for rec in records:
        tok = rec["token"]
        if tok in index:
            cid = int(rec.get("emoe_class_id", 6))
            by_class[cid].append(tok)

    hdr("Loading anchors…")
    anchors = np.load(anchors_path)
    ok(f"Anchors shape: {anchors.shape}")

    hdr(f"Loading trajectories + plotting (max {max_traj} per class)…")

    n_classes = anchors.shape[0]
    ncols = 3
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    rng = np.random.default_rng(seed=42)
    parse_errors = 0

    for c in range(n_classes):
        ax = axes[c]
        col = CLASS_COLORS[c % len(CLASS_COLORS)]
        name = SHORT_NAMES[c] if c < len(SHORT_NAMES) else f"class_{c}"

        tokens = by_class.get(c, [])
        if len(tokens) > max_traj:
            chosen = rng.choice(len(tokens), size=max_traj, replace=False)
            tokens = [tokens[i] for i in chosen]

        ok_count = 0
        for tok in tokens:
            entry = index[tok]
            try:
                traj_data = load_gz(entry["trajectory"])
                xy = extract_ego_trajectory(traj_data)
                if xy is None or xy.shape[0] < 2:
                    parse_errors += 1
                    continue

                # Convert to ego-init frame if values look global
                # Heuristic: if range > 1000 m it's probably global coords
                span = float(np.abs(xy).max())
                if span > 500:
                    xy = to_ego_frame(xy)

                ax.plot(xy[:, 0], xy[:, 1],
                        color=col, alpha=0.15, lw=0.8, zorder=2)
                # endpoint dot
                ax.scatter(xy[-1, 0], xy[-1, 1],
                           s=12, color=col, alpha=0.4,
                           linewidths=0, zorder=3)
                ok_count += 1

            except Exception:
                parse_errors += 1
                continue

        # Anchors
        anc = anchors[c]
        ax.scatter(anc[:, 0], anc[:, 1],
                   s=180, color="#1A1A2E", marker="*",
                   zorder=6, linewidths=0.4, edgecolors="white",
                   label=f"Anchors (Ka={anc.shape[0]})")

        # Ego origin
        ax.scatter(0, 0, s=80, color="black", marker="D", zorder=7)
        ax.annotate("", xy=(3, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        ax.set_title(f"{name}\n(n={ok_count} trajectories)",
                     fontsize=12, fontweight="bold", color=col)
        ax.set_xlabel("x  (ego-forward, m)", fontsize=10)
        ax.set_ylabel("y  (ego-left, m)", fontsize=10)
        ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

        scene_name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else f"class_{c}"
        print(f"  class {c} ({scene_name:<30s}): "
              f"{ok_count} plotted  ({len(by_class.get(c, []))} in cache)")

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    if parse_errors:
        warn(f"{parse_errors} trajectories could not be parsed "
             f"(extract_ego_trajectory returned None)")

    fig.suptitle(
        "EMoE Planner — GT Trajectory Spaghetti + Anchors  (ego frame)\n"
        "Each line = one scenario  |  dot = trajectory endpoint  |  ★ = KMeans anchor",
        fontsize=13, fontweight="bold",
    )

    out_path = out_dir / "fig4_trajectory_spaghetti.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    ok(f"Saved → {out_path}")

# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./emoe_viz")
    parser.add_argument("--mode", type=str, default="explore",
                        choices=["explore", "plot"],
                        help="explore: inspect structure  |  plot: spaghetti figure")
    parser.add_argument("--max_traj", type=int, default=300,
                        help="Max trajectories to plot per class (plot mode only)")
    args = parser.parse_args()

    if args.mode == "explore":
        explore_mode(Path(args.cache_dir), Path(args.labels_path))
    else:
        if args.anchors_path is None:
            raise ValueError("--anchors_path required for plot mode")
        plot_mode(
            Path(args.cache_dir),
            Path(args.labels_path),
            Path(args.anchors_path),
            Path(args.out_dir),
            args.max_traj,
        )

if __name__ == "__main__":
    main()
