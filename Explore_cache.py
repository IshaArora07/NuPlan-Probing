#!/usr/bin/env python3
“””
Diagnostic explorer: figure out what data is available from

1. scene_labels.jsonl   (from precompute script)
1. scene_anchors.npy    (from precompute script)
1. nuPlan cache files   (from PLUTO’s caching command)

Run from your project root:
python explore_cache_structure.py   
–labels_path  ./emoe_precomputed/scene_labels.jsonl   
–anchors_path ./emoe_precomputed/scene_anchors.npy   
–cache_dir    ./nuplan_cache   # or wherever PLUTO wrote its cache

Output tells you exactly what trajectory data is available and how to load it.
“””

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from collections import Counter

import numpy as np

# ── ANSI colours for terminal readability ─────────────────────────────────────

GRN  = “\033[92m”
YLW  = “\033[93m”
RED  = “\033[91m”
BLU  = “\033[94m”
RST  = “\033[0m”
BOLD = “\033[1m”

def ok(msg):  print(f”{GRN}  ✓ {msg}{RST}”)
def warn(msg): print(f”{YLW}  ⚠ {msg}{RST}”)
def err(msg):  print(f”{RED}  ✗ {msg}{RST}”)
def hdr(msg):  print(f”\n{BOLD}{BLU}{‘─’*60}\n  {msg}\n{‘─’*60}{RST}”)

# ──────────────────────────────────────────────────────────────────────────────

# 1. LABELS FILE

# ──────────────────────────────────────────────────────────────────────────────

def explore_labels(labels_path: Path):
hdr(“1. scene_labels.jsonl”)

```
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

# Show all keys in a record
sample = records[0]
print(f"\n  Top-level keys: {list(sample.keys())}")

if "debug" in sample:
    print(f"  debug sub-keys: {list(sample['debug'].keys())}")

# Class distribution
counts = Counter(r.get("emoe_class_name", "?") for r in records)
print("\n  Class distribution:")
for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"    {cls:<35s}: {n}")

# Check if endpoints are stored inline
has_endpoint = any("endpoint_xy" in r or "endpoint" in r for r in records[:20])
has_traj     = any("trajectory" in r or "waypoints" in r for r in records[:20])

if has_endpoint:
    ok("Labels contain endpoint_xy directly — no cache needed for scatter plot!")
else:
    warn("Labels do NOT contain endpoint_xy inline.")

if has_traj:
    ok("Labels contain full trajectory waypoints!")
else:
    warn("Labels do NOT contain full trajectories inline.")

# Travel distance stats
dists = [r.get("travel_distance_m", None) for r in records]
dists = [d for d in dists if d is not None]
if dists:
    arr = np.array(dists)
    print(f"\n  travel_distance_m stats: min={arr.min():.1f}  median={np.median(arr):.1f}  max={arr.max():.1f}")

tokens = [r.get("token") for r in records if r.get("token")]
print(f"\n  Sample tokens (first 3): {tokens[:3]}")

return records, counts
```

# ──────────────────────────────────────────────────────────────────────────────

# 2. ANCHORS FILE

# ──────────────────────────────────────────────────────────────────────────────

def explore_anchors(anchors_path: Path):
hdr(“2. scene_anchors.npy”)

```
if not anchors_path.exists():
    err(f"Not found: {anchors_path}")
    return None

anchors = np.load(anchors_path)
ok(f"Loaded anchors. Shape: {anchors.shape}  dtype: {anchors.dtype}")

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
    name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else f"class_{c}"
    spread = pts.std(axis=0)
    flag = f"{RED}ALL ZERO!{RST}" if all_zero else (f"{YLW}low spread{RST}" if spread.max() < 0.5 else f"{GRN}ok{RST}")
    print(f"  class {c} ({name:<30s}): unique={uniq:2d}  std=({spread[0]:.2f},{spread[1]:.2f})  {flag}")

return anchors
```

# ──────────────────────────────────────────────────────────────────────────────

# 3. CACHE DIRECTORY

# ──────────────────────────────────────────────────────────────────────────────

def explore_cache(cache_dir: Path, sample_tokens: list):
hdr(“3. nuPlan cache directory”)

```
if not cache_dir.exists():
    err(f"Not found: {cache_dir}")
    return

# Walk first 3 levels and collect file types
extensions = Counter()
all_files = []
for root, dirs, files in os.walk(cache_dir):
    # Limit depth
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

ok(f"Found {len(all_files)} files (capped at 5000)")
print(f"  File extensions: {dict(extensions.most_common(10))}")

# Show directory tree (2 levels)
print("\n  Directory structure (depth ≤ 2):")
shown = set()
for p in sorted(all_files)[:200]:
    rel = p.relative_to(cache_dir)
    parts = rel.parts
    for d in range(1, min(3, len(parts))):
        key = parts[:d]
        if key not in shown:
            shown.add(key)
            indent = "  " * d
            print(f"  {indent}{parts[d-1]}/")
    # Show the actual file at last level
    indent = "  " * len(parts)

# Try to find files matching sample tokens
if sample_tokens:
    hdr("3b. Token → cache file lookup")
    for tok in sample_tokens[:5]:
        matches = [p for p in all_files if tok in p.name or tok in str(p)]
        if matches:
            ok(f"Token {tok[:16]}... → {matches[0]}")
        else:
            warn(f"Token {tok[:16]}... → NOT FOUND in cache file names")

# Try to load a sample cache file and inspect its contents
hdr("3c. Sample cache file contents")
candidates = [p for p in all_files if p.suffix in (".npz", ".pkl", ".pt", ".npy", ".h5", ".hdf5", ".lz4", ".gz")]
if not candidates:
    warn("No known binary cache files found (.npz/.pkl/.pt/.npy/.h5/.lz4)")
    # Try any file
    candidates = all_files[:5]

for fp in candidates[:3]:
    print(f"\n  Trying: {fp.name}  ({fp.stat().st_size // 1024} KB)")
    _try_load_cache_file(fp)
```

def _try_load_cache_file(fp: Path):
suffix = fp.suffix.lower()

```
# ── numpy ──────────────────────────────────────────────────────────────────
if suffix in (".npy", ".npz"):
    try:
        data = np.load(fp, allow_pickle=True)
        if suffix == ".npz":
            ok(f"  npz keys: {list(data.keys())}")
            for k in list(data.keys())[:5]:
                arr = data[k]
                print(f"    [{k}]: shape={arr.shape}  dtype={arr.dtype}")
        else:
            print(f"    npy shape={data.shape}  dtype={data.dtype}")
    except Exception as e:
        err(f"  numpy load failed: {e}")

# ── pickle ─────────────────────────────────────────────────────────────────
elif suffix == ".pkl":
    try:
        import pickle
        with open(fp, "rb") as f:
            data = pickle.load(f)
        _print_nested(data, indent=4)
    except Exception as e:
        err(f"  pickle load failed: {e}")

# ── pytorch ────────────────────────────────────────────────────────────────
elif suffix == ".pt":
    try:
        import torch
        data = torch.load(fp, map_location="cpu", weights_only=False)
        _print_nested(data, indent=4)
    except Exception as e:
        err(f"  torch load failed: {e}")

# ── lz4 (PLUTO often uses lz4-compressed pickle) ──────────────────────────
elif suffix == ".lz4":
    try:
        import lz4.frame
        import pickle
        with lz4.frame.open(fp, "rb") as f:
            data = pickle.load(f)
        _print_nested(data, indent=4)
    except Exception as e:
        err(f"  lz4+pickle load failed: {e}")

# ── gzip pickle ────────────────────────────────────────────────────────────
elif suffix == ".gz":
    try:
        import gzip, pickle
        with gzip.open(fp, "rb") as f:
            data = pickle.load(f)
        _print_nested(data, indent=4)
    except Exception as e:
        err(f"  gzip+pickle load failed: {e}")

# ── HDF5 ──────────────────────────────────────────────────────────────────
elif suffix in (".h5", ".hdf5"):
    try:
        import h5py
        with h5py.File(fp, "r") as f:
            _print_h5(f, indent=4)
    except Exception as e:
        err(f"  h5py load failed: {e}")

else:
    warn(f"  Unknown extension {suffix}, skipping.")
```

def _print_nested(data, indent=0, max_depth=4, depth=0):
“”“Recursively print structure of dicts/lists/tensors/arrays.”””
if depth > max_depth:
print(” “ * indent + “…”)
return
pad = “ “ * indent
if isinstance(data, dict):
print(f”{pad}dict with {len(data)} keys:”)
for k in list(data.keys())[:10]:
v = data[k]
print(f”{pad}  [{k!r}]:”, end=” “)
_print_nested(v, indent + 4, max_depth, depth + 1)
elif isinstance(data, (list, tuple)):
print(f”{pad}{type(data).**name**} len={len(data)}”)
if data:
print(f”{pad}  [0]:”, end=” “)
_print_nested(data[0], indent + 4, max_depth, depth + 1)
elif hasattr(data, “shape”) and hasattr(data, “dtype”):  # np or torch
print(f”shape={data.shape}  dtype={data.dtype}”)
else:
s = str(data)
print(s[:80] + (”…” if len(s) > 80 else “”))

def _print_h5(node, indent=0, max_depth=3, depth=0):
import h5py
pad = “ “ * indent
if depth > max_depth:
print(pad + “…”)
return
if isinstance(node, h5py.File) or isinstance(node, h5py.Group):
for k in list(node.keys())[:10]:
child = node[k]
print(f”{pad}[{k}]”, end=” “)
if isinstance(child, h5py.Dataset):
print(f”dataset shape={child.shape}  dtype={child.dtype}”)
else:
print(“group”)
_print_h5(child, indent + 4, max_depth, depth + 1)

# ──────────────────────────────────────────────────────────────────────────────

# MAIN

# ──────────────────────────────────────────────────────────────────────────────

def main():
parser = argparse.ArgumentParser(description=“Explore EMoE label/anchor/cache structure”)
parser.add_argument(”–labels_path”,  type=str, required=True,
help=“Path to scene_labels.jsonl”)
parser.add_argument(”–anchors_path”, type=str, required=True,
help=“Path to scene_anchors.npy”)
parser.add_argument(”–cache_dir”,    type=str, default=None,
help=“Path to nuPlan cache directory (from PLUTO caching command)”)
args = parser.parse_args()

```
records, counts = explore_labels(Path(args.labels_path))
anchors          = explore_anchors(Path(args.anchors_path))

if args.cache_dir:
    sample_tokens = [r["token"] for r in records[:10] if "token" in r]
    explore_cache(Path(args.cache_dir), sample_tokens)
else:
    hdr("3. nuPlan cache directory")
    warn("--cache_dir not provided, skipping cache exploration.")
    print("  Re-run with --cache_dir <path> to inspect cache files.")

# ── Final summary + recommendation ────────────────────────────────────────
hdr("SUMMARY & RECOMMENDATION")

has_anchors = anchors is not None
has_labels  = bool(records)

if has_labels and has_anchors:
    ok("You have labels + anchors → scatter plot of endpoints vs anchors is READY.")
    print(f"""
```

Scatter plot script will:
• Read endpoints from labels (recompute ego_endpoint_in_ego_frame from debug geometry,
OR re-derive from cache if full trajectories are available)
• Overlay anchors[class_id] from scene_anchors.npy
• One subplot per class, ~500-1000 points each

NOTE: scene_labels.jsonl stores travel_distance_m and debug geometry scalars,
but NOT the raw (x,y) endpoint array. Two options:
A) Re-derive endpoint from debug: needs delta_heading + dist (approximate only)
B) Re-run ego_endpoint_in_ego_frame() from cache trajectories (exact, preferred)
C) Add endpoint_xy saving to your precompute script (1-line fix, cleanest)

→ Recommend option C: add this to your precompute script’s record dict:
“endpoint_xy”: endpoint_xy.tolist()   # already computed, just not saved!
“””)
else:
warn(“Missing labels or anchors — check paths.”)

if **name** == “**main**”:
main()
