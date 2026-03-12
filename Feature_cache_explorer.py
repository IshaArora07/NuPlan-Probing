#!/usr/bin/env python3
"""
Deep cache explorer — finds where emoe_class_id lives inside features.gz.

Usage:
python explore_cache_deep.py --cache_dir ./nuplan_cache --n_samples 5
"""

import gzip
import pickle
import argparse
from pathlib import Path


def recursive_print(obj, prefix="", max_depth=6, depth=0):
    """Recursively print structure of any nested object."""

    indent = "  " * depth

    if depth > max_depth:
        print(f"{indent}{prefix}... (max depth reached)")
        return

    if isinstance(obj, dict):

        print(f"{indent}{prefix}dict  keys={list(obj.keys())}")

        for k, v in obj.items():
            recursive_print(v, prefix=f"['{k}'] → ", max_depth=max_depth, depth=depth + 1)

    elif hasattr(obj, "__dict__") and not isinstance(obj, type):

        cls = type(obj).__name__

        attrs = {k: v for k, v in vars(obj).items() if not k.startswith("__")}

        print(f"{indent}{prefix}{cls} instance  attrs={list(attrs.keys())}")

        if "data" in attrs:
            recursive_print(attrs["data"], prefix=".data → ", max_depth=max_depth, depth=depth + 1)
        else:
            for k, v in attrs.items():
                recursive_print(v, prefix=f".{k} → ", max_depth=max_depth, depth=depth + 1)

    elif hasattr(obj, "keys"):

        print(f"{indent}{prefix}{type(obj).__name__}  keys={list(obj.keys())}")

    else:

        import numpy as np

        if hasattr(obj, "shape"):

            print(
                f"{indent}{prefix}{type(obj).__name__}  shape={obj.shape}  dtype={getattr(obj, 'dtype', '?')}"
            )

        elif isinstance(obj, (list, tuple)):

            print(f"{indent}{prefix}{type(obj).__name__}  len={len(obj)}")

            if len(obj) > 0:
                recursive_print(obj[0], prefix="[0] → ", max_depth=max_depth, depth=depth + 1)

        else:

            print(f"{indent}{prefix}{type(obj).__name__}  val={repr(obj)[:120]}")


def find_emoe(obj, path="root", results=None):
    """
    Recursively search for any key containing 'emoe' anywhere in the structure.
    """

    if results is None:
        results = []

    if isinstance(obj, dict):

        for k, v in obj.items():

            full_path = f"{path}['{k}']"

            if "emoe" in str(k).lower():
                results.append((full_path, v))

            find_emoe(v, full_path, results)

    elif hasattr(obj, "__dict__") and not isinstance(obj, type):

        for k, v in vars(obj).items():

            if k.startswith("__"):
                continue

            full_path = f"{path}.{k}"

            if "emoe" in str(k).lower():
                results.append((full_path, v))

            find_emoe(v, full_path, results)

    elif isinstance(obj, (list, tuple)) and len(obj) > 0:

        find_emoe(obj[0], f"{path}[0]", results)

    return results


def load_and_explore(feat_path: Path):

    print(f"\n{'='*70}")
    print(f"  File: {feat_path}")
    print(f"{'='*70}")

    with gzip.open(feat_path, "rb") as f:
        obj = pickle.load(f)

    print(f"\n[1] Top-level type: {type(obj).__name__}")

    print(f"\n[2] Searching for 'emoe' anywhere in structure...")

    hits = find_emoe(obj)

    if hits:

        for path, val in hits:
            print(f"  FOUND  {path}  →  {repr(val)[:200]}")

    else:

        print("  ✗ No 'emoe' key found anywhere")

    print(f"\n[3] Full structure:")

    recursive_print(obj, max_depth=5)

    print(f"\n[4] Trying common unwrap paths...")

    try:

        inner = obj["data"]

        if hasattr(inner, "data"):
            inner = inner.data

        emoe = inner.get("emoe") if isinstance(inner, dict) else None

        if emoe is not None:

            print(f"  ✓ obj['data'].data['emoe'] = {emoe}")

        else:

            print("  - obj['data'].data has no 'emoe' key")

            print(f"    keys = {list(inner.keys()) if isinstance(inner, dict) else '?'}")

    except Exception as e:

        print(f"  - path A failed: {e}")

    try:

        inner = obj["data"]

        if isinstance(inner, dict):

            emoe = inner.get("emoe")

            if emoe is not None:

                print(f"  ✓ obj['data']['emoe'] = {emoe}")

            else:

                print("  - obj['data'] is dict but no 'emoe' key")

                print(f"    keys = {list(inner.keys())}")

    except Exception as e:

        print(f"  - path B failed: {e}")

    try:

        if hasattr(obj, "data") and isinstance(obj.data, dict):

            emoe = obj.data.get("emoe")

            if emoe is not None:

                print(f"  ✓ obj.data['emoe'] = {emoe}")

            else:

                print("  - obj.data is dict but no 'emoe' key")

                print(f"    keys = {list(obj.data.keys())}")

    except Exception as e:

        print(f"  - path C failed: {e}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of features.gz files to inspect",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Inspect a specific token (optional)",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.token:

        matches = list(cache_dir.glob(f"*/*/{args.token}/features.gz"))

        if not matches:

            print(f"Token {args.token} not found in cache")

            return

        for p in matches:
            load_and_explore(p)

        return

    found = 0

    for log_dir in cache_dir.iterdir():

        if not log_dir.is_dir():
            continue

        for tag_dir in log_dir.iterdir():

            if not tag_dir.is_dir():
                continue

            for tok_dir in tag_dir.iterdir():

                feat_p = tok_dir / "features.gz"

                if feat_p.exists():

                    load_and_explore(feat_p)

                    found += 1

                    if found >= args.n_samples:
                        return

    if found == 0:

        print("No features.gz files found in cache_dir")


if __name__ == "__main__":
    main()
