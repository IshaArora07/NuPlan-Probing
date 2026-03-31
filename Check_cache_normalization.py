#!/usr/bin/env python3
"""
check_cache_normalization.py

Run with:
python check_cache_normalization.py --cache_path /your/cache/path

This checks whether your cached features.gz files are in global or
ego-relative coordinates, and whether 'target', 'origin', 'angle'
keys exist — which determines the correct fix for your metric bug.
"""

import argparse
import gzip
import os
import pickle

import numpy as np


def unwrap_cache(data):
    """
    Handle both plain dict cache and wrapped {'data': ...} cache.
    """
    if isinstance(data, dict) and "data" in data:
        inner = data["data"]
        if hasattr(inner, "data"):
            inner = inner.data
        return inner
    return data


def print_tensor_stats(name, arr):
    arr = np.asarray(arr)
    print(f"\n[{name}]")
    print(f"  shape : {arr.shape}")
    print(f"  max   : {np.abs(arr).max():.4f}")
    print(f"  mean  : {np.abs(arr).mean():.4f}")

    if np.abs(arr).max() > 20:
        print("  ⚠ LARGE VALUES → likely still global / not normalized")
    else:
        print("  ✓ small values → likely ego-relative")


def check_cache(cache_path: str, max_files: int = 3):
    print(f"\nSearching for features.gz in: {cache_path}\n")

    found = 0

    for root, _, files in os.walk(cache_path):
        for file_name in files:
            if file_name != "features.gz":
                continue

            fpath = os.path.join(root, file_name)

            print(f"{'=' * 80}")
            print(f"File: {fpath}")
            print(f"{'=' * 80}")

            try:
                with gzip.open(fpath, "rb") as fh:
                    raw = pickle.load(fh)
                d = unwrap_cache(raw)
            except Exception as e:
                print(f"  ERROR loading file: {e}")
                continue

            if not isinstance(d, dict):
                print(f"  Unexpected cache format: {type(d)}")
                continue

            # ── Agent position ──────────────────────────────────────
            if "agent" in d and isinstance(d["agent"], dict):
                agent = d["agent"]
            else:
                agent = None

            if agent is not None and "position" in agent:
                print_tensor_stats("agent.position", agent["position"])
            else:
                print("\n[agent.position] KEY MISSING")

            # ── Agent target ────────────────────────────────────────
            if agent is not None and "target" in agent:
                print_tensor_stats("agent.target", agent["target"])
            else:
                print(
                    "\n[agent.target] KEY MISSING — normalize() may not have run"
                )

            # ── Origin / angle ──────────────────────────────────────
            print(f"\n[origin] exists: {'origin' in d}")
            print(f"[angle]  exists: {'angle' in d}")

            if "origin" in d:
                print(f"  origin value: {d['origin']}")
            if "angle" in d:
                print(f"  angle value : {d['angle']}")

            # ── Top-level keys ──────────────────────────────────────
            print(f"\n[top-level keys]: {list(d.keys())}")
            if agent is not None:
                print(f"[agent keys]:     {list(agent.keys())}")

            print()

            found += 1
            if found >= max_files:
                print(f"Checked {max_files} files. Stopping.")
                return

    if found == 0:
        print("No features.gz files found. Check your cache path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to your nuPlan cache directory",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=3,
        help="How many cache files to inspect (default: 3)",
    )

    args = parser.parse_args()
    check_cache(args.cache_path, args.max_files)
