#!/usr/bin/env python3
"""
inspect_cache.py

Reads one features.gz cache file and prints its full structure
and key values to determine coordinate frame state.

Run:
python inspect_cache.py --cache_path /your/cache/path
"""

import argparse
import gzip
import os
import pickle

import numpy as np


def print_structure(obj, name: str, depth: int = 0, max_depth: int = 4) -> None:
    indent = "  " * depth

    if isinstance(obj, dict):
        print(f"{indent}{name}: dict  keys={list(obj.keys())}")
        if depth < max_depth:
            for k, v in obj.items():
                print_structure(v, k, depth + 1, max_depth)

    elif isinstance(obj, np.ndarray):
        print(
            f"{indent}{name}: ndarray  shape={obj.shape}  dtype={obj.dtype}  "
            f"absmax={np.abs(obj).max():.4f}  mean={np.abs(obj).mean():.4f}"
        )
        if obj.size > 0:
            flat = obj.flatten()
            print(f"{indent}  first 4 values: {flat[:4].tolist()}")

    elif isinstance(obj, (int, float, np.floating, np.integer)):
        print(f"{indent}{name}: scalar = {obj}")

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__}  len={len(obj)}")

    else:
        print(f"{indent}{name}: {type(obj).__name__} = {str(obj)[:80]}")


def find_array(obj, key_path: str):
    """Navigate a dot-separated key path like 'data.agent.position'."""
    parts = key_path.split(".")
    cur = obj

    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None

    return cur


def inspect(cache_path: str) -> None:
    # ── Find first features.gz ────────────────────────────────
    fpath = None
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            if f == "features.gz":
                fpath = os.path.join(root, f)
                break
        if fpath:
            break

    if fpath is None:
        print(f"No features.gz found under {cache_path}")
        return

    print(f"Reading: {fpath}\n")
    with gzip.open(fpath, "rb") as fh:
        raw = pickle.load(fh)

    sep = "=" * 65

    # ── Full structure ────────────────────────────────────────
    print(sep)
    print("FULL CACHE STRUCTURE")
    print(sep)
    print(f"Top-level type: {type(raw)}")

    if hasattr(raw, "__dict__"):
        print(f"Object attrs: {list(raw.__dict__.keys())}")
        # PlutoFeature often stores data in .data
        if hasattr(raw, "data"):
            print("\nraw.data structure:")
            print_structure(raw.data, "data", depth=1)
            d = raw.data
        else:
            print("No .data attribute — using object __dict__")
            d = raw.__dict__

    elif isinstance(raw, dict):
        print_structure(raw, "root", depth=0)
        d = raw

    else:
        print(f"Unknown type: {type(raw)}")
        return

    # ── Try to find key arrays ────────────────────────────────
    print(f"\n{sep}")
    print("KEY ARRAY VALUES")
    print(sep)

    candidates = {
        "agent.position": ["agent.position", "data.agent.position"],
        "agent.heading": ["agent.heading", "data.agent.heading"],
        "agent.target": ["agent.target", "data.agent.target"],
        "map.polygon_center": ["map.polygon_center", "data.map.polygon_center"],
        "origin": ["origin", "data.origin"],
        "angle": ["angle", "data.angle"],
    }

    found = {}
    for label, paths in candidates.items():
        for path in paths:
            val = find_array(d, path)
            if val is not None:
                found[label] = val
                break

    if not found:
        print("Could not find any expected arrays. Raw structure above is all we have.")
        return

    for label, arr in found.items():
        print(f"\n[{label}]")

        if isinstance(arr, np.ndarray):
            print(f"  shape  : {arr.shape}")
            print(f"  dtype  : {arr.dtype}")
            print(f"  absmax : {np.abs(arr).max():.6f}")
            print(f"  first 4 flat values: {arr.flatten()[:4].tolist()}")

            # Specific useful slices
            if label == "agent.position" and arr.ndim >= 3:
                print(f"  ego present step (idx 20): {arr[0, 20, :]}")
                print(f"  ego first step   (idx  0): {arr[0, 0, :]}")
                if arr.shape[0] > 1:
                    print(f"  agent1 present   (idx 20): {arr[1, 20, :]}")

            if label == "agent.target" and arr.ndim >= 3:
                print(f"  ego target[:3,:2]: {arr[0, :3, :2]}")

            if label == "map.polygon_center" and arr.ndim >= 2:
                print(f"  first 3 entries: {arr[:3]}")

            if label == "origin":
                print(f"  full value: {arr}")

        else:
            print(f"  value: {arr}")

    # ── Conclusion ────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONCLUSION")
    print(sep)

    pos = found.get("agent.position")
    tgt = found.get("agent.target")
    pc = found.get("map.polygon_center")
    origin = found.get("origin")

    if pos is None:
        print("  ❌ Could not find agent.position — check structure above")
        return

    pos_absmax = np.abs(pos).max()
    ego_zero = pos.ndim >= 3 and np.abs(pos[0, 20, :]).max() < 1.0
    tgt_ok = tgt is not None and np.abs(tgt).max() < 80
    map_utm = pc is not None and np.abs(pc[..., :2]).max() > 1000
    origin_utm = origin is not None and np.abs(origin).max() > 1000

    tgt_absmax_str = f"{np.abs(tgt).max():.2f}" if tgt is not None else "N/A"

    print(f"  position absmax  : {pos_absmax:.2f}")
    print(f"  ego at origin    : {ego_zero}")
    print(f"  target correct   : {tgt_ok}  (absmax={tgt_absmax_str})")
    print(f"  map in UTM       : {map_utm}")
    print(f"  origin is UTM    : {origin_utm}")

    print()
    if pos_absmax < 200 and tgt_ok and not map_utm:
        print("  ✅ CACHE IS FULLY NORMALIZED")
        print("     Do NOT call normalize_batch.")
        print("     The metric bug is in _compute_metrics or model output frames.")

    elif pos_absmax > 1000 and not ego_zero:
        print("  ❌ CACHE IS IN RAW UTM — full normalize_batch needed")
        print("     All positions, headings, map, targets need transform.")

    elif ego_zero and pos_absmax < 200 and map_utm:
        print("  ⚠️  POSITIONS NORMALIZED but MAP IS STILL IN UTM")
        print("     normalize_batch should only transform map, not positions.")

    elif ego_zero and pos_absmax < 200 and not tgt_ok:
        print("  ⚠️  POSITIONS NORMALIZED but TARGET IS WRONG")
        print("     _recompute_target needs to run, but rot_pos should NOT run.")

    elif ego_zero and map_utm:
        print("  ⚠️  PARTIAL: ego at origin, map in UTM, other agents may be mixed")
        print("     Inspect agent1 position above to determine other agent frame.")

    else:
        print("  ⚠️  MIXED STATE — read values above carefully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to nuPlan cache directory",
    )
    args = parser.parse_args()
    inspect(args.cache_path)
