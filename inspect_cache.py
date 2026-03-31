#!/usr/bin/env python3
"""
inspect_trajectory_gz.py

Inspects the trajectory.gz cache file to determine its structure
and whether stored trajectories are in UTM or ego-relative frame.

Run:
    python inspect_trajectory_gz.py --cache_path /your/cache/path
"""

import argparse
import gzip
import os
import pickle
from typing import Any

import numpy as np


def print_recursive(
    obj: Any,
    name: str = "root",
    depth: int = 0,
    max_depth: int = 5,
) -> None:
    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}{name}: ...")
        return

    if isinstance(obj, np.ndarray):
        try:
            if np.issubdtype(obj.dtype, np.floating):
                finite = obj[np.isfinite(obj)]
            else:
                finite = obj.reshape(-1)

            amax = float(np.abs(finite).max()) if finite.size > 0 else 0.0
            first4 = obj.reshape(-1)[:4].tolist() if obj.size > 0 else []

            print(
                f"{indent}{name}: ndarray  "
                f"shape={obj.shape}  dtype={obj.dtype}  "
                f"absmax={amax:.4f}  first4={first4}"
            )
        except Exception as e:
            print(
                f"{indent}{name}: ndarray  "
                f"shape={obj.shape}  dtype={obj.dtype}  "
                f"(error inspecting: {e})"
            )

    elif isinstance(obj, dict):
        print(f"{indent}{name}: dict  keys={list(obj.keys())}")
        for k, v in obj.items():
            print_recursive(v, name=str(k), depth=depth + 1, max_depth=max_depth)

    elif hasattr(obj, "data") and not isinstance(obj, (str, bytes, np.ndarray)):
        print(f"{indent}{name}: {type(obj).__name__}  (has .data)")
        print_recursive(obj.data, name="data", depth=depth + 1, max_depth=max_depth)

    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        print(
            f"{indent}{name}: {type(obj).__name__}  "
            f"attrs={list(obj.__dict__.keys())}"
        )
        for k, v in obj.__dict__.items():
            print_recursive(v, name=str(k), depth=depth + 1, max_depth=max_depth)

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__}  len={len(obj)}")
        if len(obj) > 0:
            print_recursive(
                obj[0],
                name=f"{name}[0]",
                depth=depth + 1,
                max_depth=max_depth,
            )

    elif isinstance(obj, (int, float, bool, np.integer, np.floating, np.bool_)):
        print(f"{indent}{name}: {type(obj).__name__} = {obj}")

    else:
        print(f"{indent}{name}: {type(obj).__name__} = {str(obj)[:80]}")


def inspect_trajectory(cache_path: str) -> None:
    sep = "=" * 65

    # ── Find first trajectory.gz ──────────────────────────────
    fpath = None
    for root, _, files in os.walk(cache_path):
        for fname in files:
            if fname == "trajectory.gz":
                fpath = os.path.join(root, fname)
                break
        if fpath:
            break

    if fpath is None:
        print(f"No trajectory.gz found under: {cache_path}")
        return

    print(f"File: {fpath}\n")

    with gzip.open(fpath, "rb") as fh:
        raw = pickle.load(fh)

    # ── Full structure ────────────────────────────────────────
    print(sep)
    print("FULL STRUCTURE OF trajectory.gz")
    print(sep)
    print_recursive(raw, name="root", max_depth=5)

    # ── Compare with sibling features.gz ─────────────────────
    feat_path = os.path.join(os.path.dirname(fpath), "features.gz")
    if os.path.exists(feat_path):
        print(f"\n{sep}")
        print("FULL STRUCTURE OF features.gz (same scenario, comparison)")
        print(sep)

        with gzip.open(feat_path, "rb") as fh:
            feat = pickle.load(fh)

        print_recursive(feat, name="root", max_depth=5)

    print(f"\n{sep}")
    print("DONE — inspect the structure above")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to nuPlan cache directory",
    )
    args = parser.parse_args()
    inspect_trajectory(args.cache_path)
