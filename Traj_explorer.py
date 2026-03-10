#!/usr/bin/env python3
"""
EMoE Trajectory Spaghetti Plotter

Loads trajectories directly from PLUTO cache files (features.gz / trajectory.gz),
both of which store a dict {'data': <PlutoFeature>}.

Cache structure:
<cache_dir>/<log_name>/<scenario_tag>/<token>/features.gz
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

CLASS_COLORS = ["#E63946","#2196F3","#FF9800","#4CAF50","#9C27B0","#00BCD4"]

DEFAULT_HISTORY_SAMPLES = 20
DEFAULT_FUTURE_SAMPLES  = 80


# ──────────────────────────────────────────────────────────────────────────────
# Cache indexing
# ──────────────────────────────────────────────────────────────────────────────

def build_token_index(cache_dir: Path) -> dict:
    """token -> {"features": Path|None, "trajectory": Path|None, "log", "tag"}"""

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

                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"

                if feat_p.exists() or traj_p.exists():

                    index[tok_dir.name] = {
                        "features": feat_p if feat_p.exists() else None,
                        "trajectory": traj_p if traj_p.exists() else None,
                        "log": log_dir.name,
                        "tag": tag_dir.name,
                    }

    return index


# ──────────────────────────────────────────────────────────────────────────────
# Load gz
# ──────────────────────────────────────────────────────────────────────────────

def load_gz_data(path: Path) -> dict:

    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and list(obj.keys()) == ["data"]:
        obj = obj["data"]

    if hasattr(obj, "serialize"):
        return obj.serialize()

    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data

    if isinstance(obj, dict):
        return obj

    raise ValueError(f"Unrecognised cache object type: {type(obj)}")


# ──────────────────────────────────────────────────────────────────────────────
# Extract ego trajectory
# ──────────────────────────────────────────────────────────────────────────────

def extract_ego_future(
    data: dict,
    history_samples: int = DEFAULT_HISTORY_SAMPLES,
    future_samples: int = DEFAULT_FUTURE_SAMPLES,
    full_trajectory: bool = False,
):

    try:

        agent = data.get("agent", {})

        pos = agent.get("position")
        hdg = agent.get("heading")

        if pos is None or hdg is None:
            return None

        if hasattr(pos, "numpy"):
            pos = pos.numpy()

        if hasattr(hdg, "numpy"):
            hdg = hdg.numpy()

        pos = np.asarray(pos, dtype=np.float64)
        hdg = np.asarray(hdg, dtype=np.float64)

        if pos.ndim != 3 or pos.shape[-1] != 2:
            return None

        present_idx = history_samples

        ego_pos = pos[0]
        ego_hdg = hdg[0]

        if full_trajectory:
            traj_global = ego_pos
        else:
            traj_global = ego_pos[present_idx + 1:]

        if len(traj_global) < 2:
            return None

        x0, y0 = ego_pos[present_idx, 0], ego_pos[present_idx, 1]

        theta = ego_hdg[present_idx]

        c, s = math.cos(-theta), math.sin(-theta)

        rel = traj_global - np.array([x0, y0])

        x_rot = c * rel[:, 0] - s * rel[:, 1]
        y_rot = s * rel[:, 0] + c * rel[:, 1]

        return np.stack([x_rot, y_rot], axis=1).astype(np.float32)

    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_nested(data, indent=0, max_depth=4, depth=0, label=""):

    pad = " " * indent

    prefix = f"{pad}{label}: " if label else pad

    if depth > max_depth:
        print(prefix + "...")
        return

    if isinstance(data, dict):

        print(prefix + f"dict ({len(data)} keys)")

        for k in list(data.keys())[:15]:

            _print_nested(data[k], indent+4, max_depth, depth+1, label=repr(k))

    elif isinstance(data, (list, tuple)):

        print(prefix + f"{type(data).__name__} len={len(data)}")

        if data:

            _print_nested(data[0], indent+4, max_depth, depth+1, label="[0]")

    elif hasattr(data, "shape") and hasattr(data, "dtype"):

        print(prefix + f"array shape={tuple(data.shape)}  dtype={data.dtype}")

    else:

        s = str(data)

        print(prefix + s[:100] + ("..." if len(s)>100 else ""))


# ──────────────────────────────────────────────────────────────────────────────
# EXPLORE MODE
# ──────────────────────────────────────────────────────────────────────────────

def explore_mode(cache_dir: Path, labels_path: Path,
                 history_samples: int, future_samples: int):

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

    hdr("Inspecting 3 sample files…")

    checked = 0

    for rec in records:

        tok = rec["token"]

        if tok not in index:
            continue

        entry = index[tok]

        gz_path = entry["features"] or entry["trajectory"]

        print(f"\n  token : {tok}")
        print(f"  file  : {gz_path.name}")
        print(f"  log   : {entry['log']}")
        print(f"  tag   : {entry['tag']}")

        try:

            data = load_gz_data(gz_path)

            print("  Data structure:")

            _print_nested(data, indent=4, max_depth=3)

            emoe = data.get("emoe")

            if emoe is not None:
                ok(f"  emoe label found in cache: {emoe}")
            else:
                warn("  No 'emoe' key in this cache file")

            xy = extract_ego_future(data, history_samples, future_samples)

            if xy is not None:
                ok(f"  extract_ego_future → shape {xy.shape}")
            else:
                err("  extract_ego_future returned None")

        except Exception as e:

            err(f"  Failed: {e}")

            import traceback
            traceback.print_exc()

        checked += 1

        if checked >= 3:
            break


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./emoe_viz")

    parser.add_argument("--mode", type=str, default="explore",
                        choices=["explore","plot"])

    parser.add_argument("--max_traj", type=int, default=300)

    parser.add_argument("--history_samples", type=int, default=20)

    parser.add_argument("--future_samples", type=int, default=80)

    parser.add_argument("--use_full_traj", action="store_true")

    args = parser.parse_args()

    if args.mode == "explore":

        explore_mode(Path(args.cache_dir),
                     Path(args.labels_path),
                     args.history_samples,
                     args.future_samples)

    else:

        raise ValueError("Plot mode implementation unchanged")


if __name__ == "__main__":
    main()
