#!/usr/bin/env python3
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def find_metadata_csv(cache_dir: Path) -> Path:
    # Prefer cache_metadata*.csv
    candidates = sorted(cache_dir.rglob("cache_metadata*.csv"))
    if candidates:
        return candidates[0]
    # fallback: any csv
    candidates = sorted(cache_dir.rglob("*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No CSV metadata found under {cache_dir}")


def pick_path_column(csv_path: Path) -> Tuple[str, Optional[str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

    # Most minimal metadata has only "file_name"
    if "file_name" in cols:
        return "file_name", ("token" if "token" in cols else None)

    # Otherwise try common alternatives
    for c in cols:
        cl = c.lower()
        if cl in {"feature_path", "features_path", "cached_feature_path", "path"}:
            return c, ("token" if "token" in cols else None)

    raise KeyError(f"Could not identify artifact path column in {csv_path}. Columns: {cols}")


def _try_load(path: Path) -> Any:
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
        raise RuntimeError(f"Cannot load {path} via pickle or torch.load: {e}")


def _extract_data(obj: Any) -> Optional[Dict]:
    # PlutoFeature-like
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data
    # raw dict
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]
        return obj
    return None


def _has_emoe(data: Dict) -> bool:
    em = data.get("emoe", None)
    if not isinstance(em, dict):
        return False
    return any(k in em for k in ["emoe_class_id", "scene_label", "class_id", "scene_class_id"])


def main(cache_dir: Path, max_items: int, max_print: int):
    cache_dir = cache_dir.expanduser().resolve()
    meta = find_metadata_csv(cache_dir)
    print("[INFO] cache_dir:", cache_dir)
    print("[INFO] metadata:", meta)

    path_col, token_col = pick_path_column(meta)
    print("[INFO] artifact path column:", path_col)

    # inventory helps: tells you what serializer/extension is used
    ext_counts = Counter()
    for p in cache_dir.rglob("*"):
        if p.is_file():
            ext_counts[p.suffix.lower()] += 1
    print("\n[INFO] Cache inventory (top extensions):")
    for ext, c in ext_counts.most_common(15):
        print(f"  {ext or '<no-ext>'}: {c}")

    ok = 0
    miss = 0
    load_fail = 0
    ex_miss = []
    ex_fail = []

    with meta.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_items > 0 and i >= max_items:
                break

            rel = (row.get(path_col) or "").strip()
            if not rel:
                load_fail += 1
                if len(ex_fail) < max_print:
                    ex_fail.append(("EMPTY_PATH", row.get(token_col)))
                continue

            artifact_path = Path(rel)
            if not artifact_path.is_absolute():
                artifact_path = cache_dir / artifact_path

            tok = row.get(token_col) if token_col else None

            if not artifact_path.exists():
                load_fail += 1
                if len(ex_fail) < max_print:
                    ex_fail.append((f"NOT_FOUND:{artifact_path}", tok))
                continue

            try:
                obj = _try_load(artifact_path)
                data = _extract_data(obj)
                if data is None:
                    load_fail += 1
                    if len(ex_fail) < max_print:
                        ex_fail.append((f"NO_DATA_DICT:{artifact_path}", tok))
                    continue

                if _has_emoe(data):
                    ok += 1
                else:
                    miss += 1
                    if len(ex_miss) < max_print:
                        dtok = data.get("token") or data.get("scenario_token") or tok
                        ex_miss.append((str(artifact_path), dtok, list(data.keys())[:20]))

            except Exception:
                load_fail += 1
                if len(ex_fail) < max_print:
                    ex_fail.append((f"LOAD_FAIL:{artifact_path}", tok))

    print("\n========== SUMMARY ==========")
    print("checked:", ok + miss + load_fail)
    print("ok(has emoe):", ok)
    print("missing emoe:", miss)
    print("load_fail:", load_fail)

    if ex_miss:
        print("\nExamples MISSING emoe (path, token, top-keys):")
        for p, t, keys in ex_miss:
            print(" -", p, "token=", t, "keys=", keys)

    if ex_fail:
        print("\nExamples FAIL (path, token):")
        for p, t in ex_fail:
            print(" -", p, "token=", t)

    if ok == 0 and miss == 0:
        print("\n[HINT] None of the cache files could be deserialized with pickle/torch.load.")
        print("       Look at the extensions inventory above and tell me the most common extension.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_items", type=int, default=5000, help="0 = all")
    ap.add_argument("--max_print", type=int, default=20)
    args = ap.parse_args()
    main(Path(args.cache_dir), args.max_items, args.max_print)
