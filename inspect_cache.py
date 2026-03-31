#!/usr/bin/env python3
"""
inspect_cache.py

Safely inspects a nuPlan features.gz cache file without assuming
any particular structure. Prints everything it finds.

Run:
python inspect_cache.py --cache_path /your/cache/path
"""

import argparse
import gzip
import os
import pickle

import numpy as np


def safe_print_dict(d, prefix="", max_depth=5, depth=0):
    if depth > max_depth:
        return

    if isinstance(d, dict):
        for k, v in d.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            safe_print_dict(v, prefix=next_prefix, max_depth=max_depth, depth=depth + 1)

    elif isinstance(d, np.ndarray):
        try:
            if d.size > 0:
                finite = d[np.isfinite(d)]
                amax = float(np.abs(finite).max()) if finite.size > 0 else 0.0
                first = d.flatten()[:4].tolist()
            else:
                amax = 0.0
                first = []

            print(
                f"  [{prefix}]  shape={d.shape}  dtype={d.dtype}  "
                f"absmax={amax:.4f}  first4={first}"
            )
        except Exception as e:
            print(f"  [{prefix}]  shape={d.shape}  (error reading values: {e})")

    elif isinstance(d, (int, float, np.floating, np.integer, bool, np.bool_)):
        print(f"  [{prefix}]  scalar = {d}")

    elif isinstance(d, (list, tuple)):
        print(f"  [{prefix}]  {type(d).__name__} len={len(d)}")
        if len(d) > 0:
            safe_print_dict(
                d[0],
                prefix=f"{prefix}[0]",
                max_depth=max_depth,
                depth=depth + 1,
            )

    else:
        print(f"  [{prefix}]  {type(d).__name__} = {str(d)[:60]}")


def get_nested(obj, *keys):
    """Safely get nested key. Returns None if any key missing."""
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif hasattr(cur, k):
            cur = getattr(cur, k)
        else:
            return None
    return cur


def inspect(cache_path: str) -> None:
    sep = "=" * 65

    # ── Find first features.gz ────────────────────────────────
    fpath = None
    for root, dirs, files in os.walk(cache_path):
        for fname in files:
            if fname == "features.gz":
                fpath = os.path.join(root, fname)
                break
        if fpath:
            break

    if fpath is None:
        print(f"No features.gz found under: {cache_path}")
        return

    print(f"File: {fpath}\n")

    with gzip.open(fpath, "rb") as fh:
        raw = pickle.load(fh)

    # ── Determine root dict ───────────────────────────────────
    print(sep)
    print("OBJECT TYPE:", type(raw))

    d = None
    if isinstance(raw, dict):
        print("Structure: plain dict")
        d = raw

    elif hasattr(raw, "data") and isinstance(raw.data, dict):
        print("Structure: object with .data dict")
        d = raw.data

    elif hasattr(raw, "__dict__"):
        print("Structure: object with __dict__")
        d = raw.__dict__
        if "data" in d and isinstance(d["data"], dict):
            print("  → using .data sub-dict")
            d = d["data"]

    else:
        print("Unknown structure — printing raw repr:")
        print(repr(raw)[:500])
        return

    print(f"Top-level keys: {list(d.keys())}")
    print(sep)

    # ── Print everything ──────────────────────────────────────
    print("\nFULL STRUCTURE (all arrays with shapes and absmax):")
    safe_print_dict(d)

    # ── Targeted checks ───────────────────────────────────────
    print(f"\n{sep}")
    print("TARGETED VALUE CHECKS")
    print(sep)

    # agent.position
    pos_paths = [
        ("agent", "position"),
        ("data", "agent", "position"),
    ]
    pos = None
    for path in pos_paths:
        pos = get_nested(d, *path)
        if pos is not None:
            print(f"agent.position found at path: {'.'.join(path)}")
            break

    if pos is None:
        print("agent.position: NOT FOUND — see structure above for correct path")

    if pos is not None and isinstance(pos, np.ndarray) and pos.ndim >= 3:
        print(f"  shape           : {pos.shape}")
        print(f"  absmax          : {np.abs(pos).max():.4f}")
        idx = min(20, pos.shape[1] - 1)
        print(f"  ego at t={idx}    : {pos[0, idx, :]}")
        print(f"  ego at t=0      : {pos[0, 0, :]}")
        if pos.shape[0] > 1:
            print(f"  agent1 at t={idx}  : {pos[1, idx, :]}")

    # agent.target
    tgt_paths = [
        ("agent", "target"),
        ("data", "agent", "target"),
    ]
    tgt = None
    for path in tgt_paths:
        tgt = get_nested(d, *path)
        if tgt is not None:
            print(f"\nagent.target found at path: {'.'.join(path)}")
            break

    if tgt is None:
        print("\nagent.target: NOT FOUND")
    elif isinstance(tgt, np.ndarray):
        print(f"  shape  : {tgt.shape}")
        print(f"  absmax : {np.abs(tgt).max():.4f}")
        if tgt.ndim >= 3:
            print(f"  ego target[:3,:2]: {tgt[0, :3, :2]}")

    # map.polygon_center
    pc_paths = [
        ("map", "polygon_center"),
        ("data", "map", "polygon_center"),
    ]
    pc = None
    for path in pc_paths:
        pc = get_nested(d, *path)
        if pc is not None:
            print(f"\nmap.polygon_center found at path: {'.'.join(path)}")
            break

    if pc is None:
        print("\nmap.polygon_center: NOT FOUND")
    elif isinstance(pc, np.ndarray):
        print(f"  shape     : {pc.shape}")
        print(f"  absmax xy : {np.abs(pc[..., :2]).max():.4f}")
        print(f"  first entry: {pc[0]}")

    # origin / angle
    for key in ("origin", "angle"):
        val = d.get(key, None) if isinstance(d, dict) else None
        if val is None:
            val = get_nested(d, "data", key)

        if val is not None:
            print(f"\n{key}: {val}")
        else:
            print(f"\n{key}: NOT FOUND")

    # ── Conclusion ────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONCLUSION")
    print(sep)

    if pos is None:
        print("Cannot conclude — agent.position not found. Read structure above.")
        return

    pos_max = float(np.abs(pos).max())

    if pos.ndim >= 3:
        idx = min(20, pos.shape[1] - 1)
        ego_zero = float(np.abs(pos[0, idx, :]).max()) < 1.0
    else:
        ego_zero = False

    tgt_absmax = float(np.abs(tgt).max()) if tgt is not None else None
    tgt_ok = tgt_absmax is not None and tgt_absmax < 80

    map_utm = (
        pc is not None
        and isinstance(pc, np.ndarray)
        and float(np.abs(pc[..., :2]).max()) > 1000
    )

    print(f"  pos absmax      : {pos_max:.2f}")
    print(f"  ego at origin   : {ego_zero}")
    print(f"  target ok (<80) : {tgt_ok}   absmax={tgt_absmax if tgt_absmax is not None else 'N/A'}")
    print(f"  map in UTM      : {map_utm}")

    print()
    if pos_max < 200 and tgt_ok and not map_utm:
        print("  ✅ CACHE FULLY NORMALIZED — do NOT use normalize_batch")
        print("     Metric bug is elsewhere (model output frame or _compute_metrics)")

    elif pos_max > 1000:
        print("  ❌ CACHE IN RAW UTM — full normalize_batch needed")

    elif ego_zero and pos_max < 200 and map_utm:
        print("  ⚠️  POSITIONS OK, MAP STILL IN UTM")
        print("     normalize_batch should only transform map, not agent positions")

    elif ego_zero and pos_max < 200 and not tgt_ok:
        print("  ⚠️  POSITIONS OK, TARGET WRONG")
        print("     Only _recompute_target needed, not full normalize_batch")

    else:
        print("  ⚠️  MIXED — read values above")

    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    args = parser.parse_args()
    inspect(args.cache_path)
