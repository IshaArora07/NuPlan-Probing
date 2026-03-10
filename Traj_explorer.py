#!/usr/bin/env python3
"""
EMoE Trajectory Spaghetti Plotter

Loads GT trajectories from PLUTO cache:
trajectory.gz  →  {'data': ndarray shape (8, 3)}
8 waypoints at 1s intervals (0→8s future)
columns: (x, y, heading)  in ego-init frame already

Cache structure:
<cache_dir>/<log_name>/<scenario_tag>/<token>/trajectory.gz
"""

import gzip
import json
import pickle
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── ANSI colours ──────────────────────────────────────────────────────────────

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
# Load trajectory.gz
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectory(path: Path):

    try:

        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            arr = obj.get("data", obj)
        else:
            arr = obj

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

# ──────────────────────────────────────────────────────────────────────────────
# EXPLORE MODE
# ──────────────────────────────────────────────────────────────────────────────

def explore_mode(cache_dir: Path, labels_path: Path):

    hdr("Building token index…")

    index = build_token_index(cache_dir)

    ok(f"Indexed {len(index)} tokens")

    hdr("Checking label tokens against index")

    records = []

    with labels_path.open() as f:

        for line in f:

            line = line.strip()

            if line:

                try:
                    records.append(json.loads(line))
                except:
                    pass

    found = sum(1 for r in records if r["token"] in index)

    ok(f"{found} / {len(records)} label tokens found in cache")

    hdr("Inspecting 3 sample trajectory.gz files…")

    checked = 0

    for rec in records:

        tok = rec["token"]

        if tok not in index:
            continue

        entry = index[tok]

        if entry["trajectory"] is None:
            continue

        print(f"\n  token : {tok}")
        print(f"  class : {rec.get('emoe_class_name', '?')}  (id={rec.get('emoe_class_id', '?')})")
        print(f"  log   : {entry['log']}")
        print(f"  tag   : {entry['tag']}")

        try:

            with gzip.open(entry["trajectory"], "rb") as f:
                raw = pickle.load(f)

            print(f"  raw type  : {type(raw)}")

            if isinstance(raw, dict):

                print(f"  raw keys  : {list(raw.keys())}")

                inner = raw.get("data", None)

                if inner is not None:

                    if hasattr(inner, "shape"):
                        print(f"  data shape: {inner.shape}  dtype={inner.dtype}")
                        print(f"  data[:3]  :\n{np.asarray(inner[:3])}")
                    else:
                        print(f"  data type : {type(inner)}")

            xy = load_trajectory(entry["trajectory"])

            if xy is not None:

                ok(f"  load_trajectory → shape {xy.shape}  endpoint=({xy[-1,0]:.2f}, {xy[-1,1]:.2f}) m")

            else:

                err("  load_trajectory returned None")

        except Exception as e:

            err(f"  Failed: {e}")

            import traceback
            traceback.print_exc()

        checked += 1

        if checked >= 3:
            break

    hdr("Summary")

    print(f"  Cached tokens : {len(index)}")
    print(f"  Labels found  : {found} / {len(records)}")

    print("\nIf load_trajectory printed a valid shape → run --mode plot\n")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT MODE
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

    token_to_class = {}

    with labels_path.open() as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                rec = json.loads(line)

                token_to_class[rec["token"]] = int(rec.get("emoe_class_id", 6))

            except:
                pass

    ok(f"Loaded {len(token_to_class)} label records")

    by_class = defaultdict(list)

    for tok, cid in token_to_class.items():

        if tok in index and index[tok]["trajectory"] is not None and cid < 6:

            by_class[cid].append(tok)

    print()

    for c in range(6):

        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:<30s}): {len(by_class[c])} tokens with trajectory.gz")

    hdr("Loading anchors…")

    anchors = np.load(anchors_path)

    ok(f"Anchors shape: {anchors.shape}")

    hdr(f"Plotting (max {max_traj} per class)…")

    rng = np.random.default_rng(seed=42)

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

            idx = rng.choice(len(tokens), size=max_traj, replace=False)

            tokens = [tokens[i] for i in idx]

        ok_count = 0
        endpoints = []

        for tok in tokens:

            traj_path = index[tok]["trajectory"]

            xy = load_trajectory(traj_path)

            if xy is None or xy.shape[0] < 2:

                parse_errors += 1

                continue

            origin = np.zeros((1, 2), dtype=np.float32)

            xy_full = np.concatenate([origin, xy], axis=0)

            ax.plot(xy_full[:, 0], xy_full[:, 1],
                    color=col, alpha=0.15, lw=0.9, zorder=2)

            ax.scatter(xy[-1, 0], xy[-1, 1],
                       s=12, color=col, alpha=0.5,
                       linewidths=0, zorder=3)

            endpoints.append(xy[-1])

            ok_count += 1

        anc = anchors[c]

        ax.scatter(anc[:, 0], anc[:, 1],
                   s=200, color="#1A1A2E", marker="*",
                   zorder=6, linewidths=0.5, edgecolors="white",
                   label=f"Anchors (Ka={anc.shape[0]})")

        ax.scatter(0, 0, s=90, color="black", marker="D", zorder=7,
                   label="Ego origin")

        ax.annotate("", xy=(3, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        if endpoints:

            ep_arr = np.stack(endpoints)

            med_dist = float(np.median(np.linalg.norm(ep_arr, axis=1)))

            title_extra = f"  |  median dist={med_dist:.1f} m"

        else:

            title_extra = ""

        ax.set_title(f"{name}\n(n={ok_count} trajectories{title_extra})",
                     fontsize=11, fontweight="bold", color=col)

        ax.set_xlabel("x  (ego-forward, m)", fontsize=10)
        ax.set_ylabel("y  (ego-left, m)", fontsize=10)

        ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)

        ax.set_aspect("equal", adjustable="datalim")

        ax.grid(True, alpha=0.18)

        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:<30s}): {ok_count} plotted")

    for i in range(6, len(axes)):
        axes[i].set_visible(False)

    if parse_errors:
        warn(f"{parse_errors} files failed to parse")

    fig.suptitle(
        "EMoE Planner — GT Trajectory Spaghetti + KMeans Anchors  (ego frame)\n"
        "Each line = 8s GT future  |  dot = 8s endpoint  |  ★ = KMeans anchor",
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
    parser.add_argument("--labels_path",  type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir",      type=str, default="./emoe_viz")

    parser.add_argument("--mode", type=str, default="explore",
                        choices=["explore", "plot"])

    parser.add_argument("--max_traj", type=int, default=300)

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
