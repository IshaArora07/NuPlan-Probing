#!/usr/bin/env python3
"""
Diagnostic explorer: figure out what data is available from

1. scene_labels.jsonl   (from precompute script)
2. scene_anchors.npy    (from precompute script)
3. nuPlan cache files   (from PLUTO’s caching command)

Run from your project root:
python explore_cache_structure.py \
    --labels_path  ./emoe_precomputed/scene_labels.jsonl \
    --anchors_path ./emoe_precomputed/scene_anchors.npy \
    --cache_dir    ./nuplan_cache

Output tells you exactly what trajectory data is available and how to load it.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

# ── ANSI colours ──────────────────────────────────────────────────────────────

GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
BLU = "\033[94m"
RST = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"{GRN} ✓ {msg}{RST}")


def warn(msg):
    print(f"{YLW} ⚠ {msg}{RST}")


def err(msg):
    print(f"{RED} ✗ {msg}{RST}")


def hdr(msg):
    print(f"\n{BOLD}{BLU}{'─'*60}\n  {msg}\n{'─'*60}{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABELS
# ─────────────────────────────────────────────────────────────────────────────

def explore_labels(labels_path: Path):
    hdr("1. scene_labels.jsonl")

    if not labels_path.exists():
        err(f"Not found: {labels_path}")
        return [], {}

    records = []
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    ok(f"Loaded {len(records)} records")

    if not records:
        err("File is empty.")
        return [], {}

    sample = records[0]

    print(f"\nTop-level keys: {list(sample.keys())}")

    if "debug" in sample:
        print(f"debug sub-keys: {list(sample['debug'].keys())}")

    counts = Counter(r.get("emoe_class_name", "?") for r in records)

    print("\nClass distribution:")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<35s}: {n}")

    has_endpoint = any("endpoint_xy" in r or "endpoint" in r for r in records[:20])
    has_traj = any("trajectory" in r or "waypoints" in r for r in records[:20])

    if has_endpoint:
        ok("Labels contain endpoint_xy directly — no cache needed")
    else:
        warn("Labels do NOT contain endpoint_xy inline")

    if has_traj:
        ok("Labels contain full trajectory waypoints")
    else:
        warn("Labels do NOT contain full trajectories")

    dists = [r.get("travel_distance_m") for r in records if r.get("travel_distance_m")]

    if dists:
        arr = np.array(dists)
        print(
            f"\ntravel_distance_m stats: "
            f"min={arr.min():.1f} median={np.median(arr):.1f} max={arr.max():.1f}"
        )

    tokens = [r.get("token") for r in records if r.get("token")]
    print(f"\nSample tokens: {tokens[:3]}")

    return records, counts


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANCHORS
# ─────────────────────────────────────────────────────────────────────────────

def explore_anchors(anchors_path: Path):
    hdr("2. scene_anchors.npy")

    if not anchors_path.exists():
        err(f"Not found: {anchors_path}")
        return None

    anchors = np.load(anchors_path)

    ok(f"Loaded anchors shape={anchors.shape} dtype={anchors.dtype}")

    EMOE_SCENE_TYPES = [
        "left_turn_at_intersection",
        "straight_at_intersection",
        "right_turn_at_intersection",
        "straight_non_intersection",
        "roundabout",
        "u_turn",
        "others",
    ]

    for c in range(anchors.shape[0]):
        pts = anchors[c]

        all_zero = np.allclose(pts, 0.0)
        uniq = np.unique(pts, axis=0).shape[0]
        spread = pts.std(axis=0)

        name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else f"class_{c}"

        if all_zero:
            flag = f"{RED}ALL ZERO{RST}"
        elif spread.max() < 0.5:
            flag = f"{YLW}low spread{RST}"
        else:
            flag = f"{GRN}ok{RST}"

        print(
            f"class {c} ({name:<30s}) "
            f"unique={uniq} std=({spread[0]:.2f},{spread[1]:.2f}) {flag}"
        )

    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# 3. CACHE
# ─────────────────────────────────────────────────────────────────────────────

def explore_cache(cache_dir: Path, sample_tokens):
    hdr("3. nuPlan cache directory")

    if not cache_dir.exists():
        err(f"Not found: {cache_dir}")
        return

    extensions = Counter()
    all_files = []

    for root, dirs, files in os.walk(cache_dir):

        depth = len(Path(root).relative_to(cache_dir).parts)

        if depth > 3:
            dirs.clear()
            continue

        for fn in files:
            p = Path(root) / fn
            extensions[p.suffix.lower()] += 1
            all_files.append(p)

        if len(all_files) > 5000:
            break

    ok(f"Found {len(all_files)} files")

    print("File extensions:", dict(extensions.most_common(10)))

    if sample_tokens:
        hdr("Token → cache lookup")

        for tok in sample_tokens[:5]:

            matches = [p for p in all_files if tok in str(p)]

            if matches:
                ok(f"{tok[:16]} → {matches[0]}")
            else:
                warn(f"{tok[:16]} → not found")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore EMoE label / anchor / cache structure"
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to scene_labels.jsonl",
    )

    parser.add_argument(
        "--anchors_path",
        type=str,
        required=True,
        help="Path to scene_anchors.npy",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to nuPlan cache directory",
    )

    args = parser.parse_args()

    records, counts = explore_labels(Path(args.labels_path))

    anchors = explore_anchors(Path(args.anchors_path))

    if args.cache_dir:
        tokens = [r["token"] for r in records[:10] if "token" in r]
        explore_cache(Path(args.cache_dir), tokens)
    else:
        hdr("3. nuPlan cache directory")
        warn("--cache_dir not provided")


if __name__ == "__main__":
    main()
