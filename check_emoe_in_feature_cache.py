#!/usr/bin/env python3
"""
Check whether nuPlan feature cache entries contain EMoE label fields.

Strategy:
  1) Locate nuPlan cache metadata CSV (e.g., cache_metadata*.csv).
  2) Read feature artifact paths from the CSV.
  3) For a sample (or all) entries, load the cached feature object/dict.
  4) Check presence of: data["emoe"]["emoe_class_id"] (or common variants).

Notes:
- nuPlan cache layout is nested (log -> scenario_type -> token -> ...). Do not rely on *.pkl at root.
- This script is conservative: it tries pickle and torch.load.
"""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def find_metadata_csv(cache_dir: Path) -> Path:
    # Most robust: search for any csv with "cache" and "metadata" in its name
    candidates = sorted(cache_dir.rglob("*.csv"))
    scored = []
    for p in candidates:
        name = p.name.lower()
        if "cache" in name and "meta" in name:
            scored.append(p)
    if scored:
        return scored[0]

    # fallback: if only one csv exists, use it
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find cache metadata CSV under {cache_dir}. "
        f"Found {len(candidates)} csv files total."
    )


def sniff_csv_columns(csv_path: Path) -> Tuple[str, Optional[str]]:
    """
    Returns (feature_col, token_col)
    feature_col: column containing path to cached features (directory or file)
    token_col: optional column containing scenario token
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    lower = {c.lower(): c for c in fieldnames}

    # common variants across forks/versions
    feature_candidates = [
        "feature_path",
        "features_path",
        "features",
        "feature",
        "cached_feature_path",
        "cache_feature_path",
        "path_to_features",
        "features_dir",
    ]
    token_candidates = ["token", "scenario_token", "scenario_id"]

    feature_col = None
    for k in feature_candidates:
        if k in lower:
            feature_col = lower[k]
            break

    token_col = None
    for k in token_candidates:
        if k in lower:
            token_col = lower[k]
            break

    if feature_col is None:
        raise KeyError(
            f"Could not identify feature-path column in {csv_path}. "
            f"Columns: {fieldnames}"
        )
    return feature_col, token_col


def _try_load_feature_obj(path: Path) -> Any:
    """
    Try to load a cached feature artifact from a path that might be:
      - a file (pickle / torch)
      - a directory containing a file
    Returns loaded object or raises.
    """
    if path.is_dir():
        # look for likely files inside
        # prefer files that include 'feature' in name
        files = sorted([p for p in path.rglob("*") if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No files under feature dir: {path}")

        def score(p: Path) -> int:
            n = p.name.lower()
            s = 0
            if "feature" in n:
                s += 10
            if n.endswith(".pkl") or n.endswith(".pickle"):
                s += 5
            if n.endswith(".pt") or n.endswith(".pth"):
                s += 4
            return -s  # sort ascending

        files = sorted(files, key=score)
        path = files[0]

    # file load attempts
    # 1) pickle
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 2) torch.load
    try:
        return torch.load(str(path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load as pickle or torch: {path} ({e})")


def _extract_data_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Extract the dict that should contain keys like agent/map/reference_line/emoe.
    Handles:
      - PlutoFeature-like objects with .data
      - dict objects directly
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        # sometimes stored as {"feature": <PlutoFeature>} or {"data": {...}}
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]
        if "feature" in obj:
            feat = obj["feature"]
            if hasattr(feat, "data") and isinstance(feat.data, dict):
                return feat.data
        return obj
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data
    return None


def _has_emoe_class_id(data: Dict[str, Any]) -> bool:
    if "emoe" not in data or not isinstance(data["emoe"], dict):
        return False
    em = data["emoe"]
    # your code used emoe_class_id; support a few variants
    return any(k in em for k in ["emoe_class_id", "scene_label", "class_id", "scene_class_id"])


def main(cache_dir: Path, max_items: int, max_print: int, verbose: bool):
    cache_dir = cache_dir.expanduser().resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    # quick inventory (helps diagnose wrong cache_dir)
    ext_counts = Counter()
    for p in cache_dir.rglob("*"):
        if p.is_file():
            ext_counts[p.suffix.lower()] += 1
    print("[INFO] Cache inventory (top extensions):")
    for ext, c in ext_counts.most_common(12):
        print(f"  {ext or '<no-ext>'}: {c}")

    meta = find_metadata_csv(cache_dir)
    print(f"\n[INFO] Using metadata CSV: {meta}")

    feature_col, token_col = sniff_csv_columns(meta)
    print(f"[INFO] Feature column: {feature_col}")
    print(f"[INFO] Token column: {token_col}")

    ok = 0
    miss = 0
    load_fail = 0
    examples_miss = []
    examples_fail = []

    with meta.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_items > 0 and i >= max_items:
                break

            raw_path = row.get(feature_col, "") or ""
            raw_path = raw_path.strip()
            if not raw_path:
                load_fail += 1
                if len(examples_fail) < max_print:
                    examples_fail.append(("EMPTY_FEATURE_PATH", row.get(token_col)))
                continue

            feat_path = Path(raw_path)
            if not feat_path.is_absolute():
                # nuPlan metadata often stores relative paths; treat them as relative to cache_dir
                feat_path = cache_dir / feat_path

            token = row.get(token_col) if token_col else None

            try:
                obj = _try_load_feature_obj(feat_path)
                data = _extract_data_dict(obj)
                if data is None or not isinstance(data, dict):
                    load_fail += 1
                    if len(examples_fail) < max_print:
                        examples_fail.append((str(feat_path), token))
                    continue

                has = _has_emoe_class_id(data)
                if has:
                    ok += 1
                else:
                    miss += 1
                    if len(examples_miss) < max_print:
                        # try pull token from data if possible
                        dtok = data.get("token") or data.get("scenario_token") or token
                        examples_miss.append((str(feat_path), dtok, list(data.keys())[:20]))

                if verbose and (i % 500 == 0):
                    print(f"[PROGRESS] checked={i} ok={ok} miss={miss} load_fail={load_fail}")

            except Exception:
                load_fail += 1
                if len(examples_fail) < max_print:
                    examples_fail.append((str(feat_path), token))
                continue

    total = ok + miss + load_fail
    print("\n========== SUMMARY ==========")
    print(f"checked: {total}")
    print(f"ok (has emoe class id): {ok}")
    print(f"missing emoe: {miss}")
    print(f"load failures: {load_fail}")

    if examples_miss:
        print("\nExamples MISSING emoe (path, token, top-keys):")
        for p, tok, keys in examples_miss:
            print(" -", p, "token=", tok, "keys=", keys)

    if examples_fail:
        print("\nExamples LOAD FAIL (path, token):")
        for p, tok in examples_fail:
            print(" -", p, "token=", tok)

    # simple exit hint
    if ok == 0 and miss == 0:
        print("\n[HINT] No entries could be loaded. Most common causes:")
        print("  - wrong --cache_dir (pointing at experiment dir, not cache dir)")
        print("  - metadata CSV points to paths not visible on this machine/mount")
        print("  - cache files are not pickle/torch serialized (need custom loader)")
    elif miss > 0:
        print("\n[CONCLUSION] Your feature cache contains entries without emoe. "
              "Those will crash any collate code that assumes f.data['emoe'] exists.")
    else:
        print("\n[CONCLUSION] All loaded cached features contained emoe. "
              "If training still sees missing emoe, the issue is in the training dataloader path, not cache content.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="nuPlan cache root (cfg.cache.cache_path)")
    ap.add_argument("--max_items", type=int, default=2000,
                    help="How many metadata rows to check (0 = all). Default 2000.")
    ap.add_argument("--max_print", type=int, default=15,
                    help="How many example failures to print.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    main(Path(args.cache_dir), args.max_items, args.max_print, args.verbose)
