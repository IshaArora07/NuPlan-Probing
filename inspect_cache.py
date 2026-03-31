#!/usr/bin/env python3
"""
inspect_cache.py

Reads one features.gz cache file and prints exact raw values
to determine what coordinate frame the cache is stored in.

Run:
python inspect_cache.py --cache_path /your/cache/path
"""

import argparse
import gzip
import os
import pickle

import numpy as np


def inspect(cache_path: str) -> None:
    # Find first features.gz
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
        d = pickle.load(fh)

    sep = "=" * 60

    # ── Top level keys ────────────────────────────────────────
    print(sep)
    print("TOP LEVEL KEYS:", list(d.keys()))
    print("AGENT KEYS    :", list(d["agent"].keys()))
    print(sep)

    # ── Origin / angle ────────────────────────────────────────
    print("\n[origin]")
    if "origin" in d:
        print(f"  value : {d['origin']}")
        print(f"  absmax: {np.abs(d['origin']).max():.4f}")
        if np.abs(d["origin"]).max() > 1000:
            print("  → UTM coordinates stored")
        else:
            print("  → near zero — origin is relative")
    else:
        print("  MISSING")

    print("\n[angle]")
    if "angle" in d:
        angle_val = d["angle"]
        if np.isscalar(angle_val):
            print(f"  value : {angle_val:.6f} rad")
        else:
            print(f"  value : {angle_val}")
    else:
        print("  MISSING")

    # ── Agent position ────────────────────────────────────────
    pos = d["agent"]["position"]  # (A, T, 2)
    print(f"\n[agent position]  shape={pos.shape}")
    print(f"  absmax (all)           : {np.abs(pos).max():.4f}")
    print(f"  ego at t=0     pos[0,0] : {pos[0, 0, :]}")
    print(f"  ego at t=20    pos[0,20]: {pos[0, 20, :]}")
    print(f"  ego at t=21    pos[0,21]: {pos[0, 21, :]}")
    if pos.shape[0] > 1:
        print(f"  agent1 at t=20 pos[1,20]: {pos[1, 20, :]}")
        print(f"  agent1 absmax           : {np.abs(pos[1]).max():.4f}")

    # ── Agent heading ─────────────────────────────────────────
    hdg = d["agent"]["heading"]  # (A, T)
    print(f"\n[agent heading]  shape={hdg.shape}")
    print(f"  ego at t=20 hdg[0,20]: {hdg[0, 20]:.6f} rad")
    print(f"  absmax: {np.abs(hdg).max():.4f}")

    # ── Agent target ─────────────────────────────────────────
    print("\n[agent target]")
    if "target" in d["agent"]:
        tgt = d["agent"]["target"]  # (A, T_future, 3)
        print(f"  shape  : {tgt.shape}")
        print(f"  absmax : {np.abs(tgt).max():.4f}")
        print("  ego target first 3 steps tgt[0,:3,:2]:")
        print(f"    {tgt[0, :3, :2]}")
        if tgt.shape[0] > 1:
            print("  agent1 target first 3 steps tgt[1,:3,:2]:")
            print(f"    {tgt[1, :3, :2]}")
        if np.abs(tgt).max() < 80:
            print("  → looks ego/agent-relative ✓")
        else:
            print("  → too large, may be UTM ✗")
    else:
        print("  MISSING — normalize() was never called on this cache")

    # ── Map ───────────────────────────────────────────────────
    print("\n[map polygon_center]")
    if "map" in d and "polygon_center" in d["map"]:
        pc = d["map"]["polygon_center"]
        print(f"  shape     : {pc.shape}")
        print(f"  absmax xy : {np.abs(pc[..., :2]).max():.4f}")
        print("  first 3 entries:")
        for i in range(min(3, len(pc))):
            print(f"    {pc[i]}")
        if np.abs(pc[..., :2]).max() > 1000:
            print("  → map is in UTM ✗")
        else:
            print("  → map looks ego-relative ✓")
    else:
        print("  MISSING")

    # ── Static objects ────────────────────────────────────────
    if "static_objects" in d and "position" in d["static_objects"]:
        so = d["static_objects"]["position"]
        print(
            f"\n[static_objects position]  "
            f"shape={so.shape}  absmax={np.abs(so).max():.4f}"
        )

    # ── Conclusion ────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONCLUSION")
    print(sep)

    origin_utm = "origin" in d and np.abs(d["origin"]).max() > 1000
    pos_utm = np.abs(pos).max() > 1000
    tgt_ok = "target" in d["agent"] and np.abs(d["agent"]["target"]).max() < 80
    map_ok = (
        "map" in d
        and "polygon_center" in d["map"]
        and np.abs(d["map"]["polygon_center"][..., :2]).max() < 300
    )
    ego_zero = np.abs(pos[0, 20, :]).max() < 1.0

    print(f"  origin stored as UTM  : {origin_utm}")
    print(f"  positions in UTM      : {pos_utm}")
    print(f"  ego at origin (t=20)  : {ego_zero}")
    print(f"  target looks correct  : {tgt_ok}")
    print(f"  map looks correct     : {map_ok}")

    if not pos_utm and tgt_ok and map_ok:
        print("\n  ✅ CACHE IS FULLY NORMALIZED — do NOT call normalize_batch")
        print("     The batch tensors should be correct as-is.")
        print("     Your metric bug must be in _compute_metrics or the model output.")
    elif pos_utm and not ego_zero:
        print("\n  ❌ CACHE IS IN RAW UTM — normalize_batch is needed")
        print("     All positions, map, targets need rotation + translation.")
    elif not pos_utm and ego_zero and not map_ok:
        print("\n  ⚠️  PARTIAL NORMALIZATION — ego is at origin but map/others are UTM")
        print("     normalize_batch needs to skip ego position but transform map + other agents.")
    elif ego_zero and not pos_utm and not tgt_ok:
        print("\n  ⚠️  POSITIONS NORMALIZED but TARGET is wrong")
        print("     target needs recomputation from normalized positions.")
    else:
        print("\n  ⚠️  MIXED STATE — inspect values above carefully")
        print(
            f"     pos_utm={pos_utm}, ego_zero={ego_zero}, "
            f"tgt_ok={tgt_ok}, map_ok={map_ok}"
        )


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
