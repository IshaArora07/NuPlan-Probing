#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def try_deserialize_gz(path: Path) -> Tuple[Optional[Any], str]:
    """
    Returns (obj, mode) where mode describes what worked.
    """
    raw = None
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read()
    except Exception as e:
        return None, f"gzip_read_fail:{type(e).__name__}"

    # 1) pickle
    try:
        obj = pickle.loads(raw)
        return obj, "pickle"
    except Exception:
        pass

    # 2) torch.load from bytes buffer
    try:
        obj = torch.load(io.BytesIO(raw), map_location="cpu")
        return obj, "torch"
    except Exception:
        pass

    # 3) json (bytes->str)
    try:
        s = raw.decode("utf-8")
        obj = json.loads(s)
        return obj, "json"
    except Exception:
        pass

    # 4) unknown
    head = raw[:8]
    return None, f"unknown_bytes_head:{head!r}"


def extract_data_dict(obj: Any) -> Optional[Dict]:
    """
    Try to get the dict that contains keys like agent/map/reference_line/emoe.
    Handles several shapes.
    """
    if obj is None:
        return None

    # PlutoFeature-like object
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data

    # Sometimes stored as dict
    if isinstance(obj, dict):
        # common wrappers
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]
        if "feature" in obj:
            feat = obj["feature"]
            if hasattr(feat, "data") and isinstance(feat.data, dict):
                return feat.data
        return obj

    return None


def has_emoe_class_id(data: Dict) -> bool:
    em = data.get("emoe", None)
    if not isinstance(em, dict):
        return False
    return any(k in em for k in ["emoe_class_id", "scene_label", "class_id", "scene_class_id"])


def parse_cache_path(feature_gz: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Given .../<log_name>/<scenario_type>/<token>/feature.gz
    return (log_name, scenario_type, token) if possible.
    """
    parts = feature_gz.parts
    # token dir is parent of feature.gz
    token_dir = feature_gz.parent
    scenario_type_dir = token_dir.parent
    log_dir = scenario_type_dir.parent

    token = token_dir.name if token_dir is not None else None
    scenario_type = scenario_type_dir.name if scenario_type_dir is not None else None
    log_name = log_dir.name if log_dir is not None else None
    return log_name, scenario_type, token


def main(cache_dir: Path, max_items: int, max_print: int):
    cache_dir = cache_dir.expanduser().resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    feature_files = sorted(cache_dir.rglob("feature.gz"))
    print(f"[INFO] Found {len(feature_files)} feature.gz files under {cache_dir}")

    if max_items > 0:
        feature_files = feature_files[:max_items]
        print(f"[INFO] Limiting to first {len(feature_files)} files (--max_items)")

    ok = 0
    miss = 0
    load_fail = 0

    mode_counts = Counter()
    fail_reasons = Counter()

    examples_miss = []
    examples_fail = []

    for i, fp in enumerate(feature_files):
        log_name, scenario_type, token = parse_cache_path(fp)

        obj, mode = try_deserialize_gz(fp)
        mode_counts[mode] += 1

        if obj is None:
            load_fail += 1
            fail_reasons[mode] += 1
            if len(examples_fail) < max_print:
                examples_fail.append((str(fp), token, scenario_type, log_name, mode))
            continue

        data = extract_data_dict(obj)
        if data is None or not isinstance(data, dict):
            load_fail += 1
            fail_reasons["no_data_dict"] += 1
            if len(examples_fail) < max_print:
                examples_fail.append((str(fp), token, scenario_type, log_name, "no_data_dict"))
            continue

        if has_emoe_class_id(data):
            ok += 1
        else:
            miss += 1
            if len(examples_miss) < max_print:
                keys = list(data.keys())[:25]
                examples_miss.append((str(fp), token, scenario_type, log_name, keys))

        if (i + 1) % 500 == 0:
            print(f"[PROGRESS] checked={i+1} ok={ok} miss={miss} fail={load_fail}")

    print("\n========== SUMMARY ==========")
    print("checked:", ok + miss + load_fail)
    print("ok(has emoe):", ok)
    print("missing emoe:", miss)
    print("load_fail:", load_fail)

    print("\n[INFO] Deserialization modes (top):")
    for k, v in mode_counts.most_common(10):
        print(f"  {k}: {v}")

    if fail_reasons:
        print("\n[INFO] Failure reasons:")
        for k, v in fail_reasons.most_common(10):
            print(f"  {k}: {v}")

    if examples_miss:
        print("\nExamples MISSING emoe (feature.gz, token, scenario_type, log_name, top-keys):")
        for fp, tok, st, lg, keys in examples_miss:
            print(" -", fp, "token=", tok, "scenario_type=", st, "log=", lg, "keys=", keys)

    if examples_fail:
        print("\nExamples LOAD FAIL (feature.gz, token, scenario_type, log_name, reason):")
        for fp, tok, st, lg, reason in examples_fail:
            print(" -", fp, "token=", tok, "scenario_type=", st, "log=", lg, "reason=", reason)

    if ok == 0 and miss == 0 and load_fail > 0:
        print("\n[NEXT] Your feature.gz payload is not pickle/torch/json.")
        print("       In that case, send me the most common 'unknown_bytes_head:...' value printed above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="cfg.cache.cache_path root")
    ap.add_argument("--max_items", type=int, default=2000, help="0 = all feature.gz files")
    ap.add_argument("--max_print", type=int, default=20)
    args = ap.parse_args()
    main(Path(args.cache_dir), args.max_items, args.max_print)
