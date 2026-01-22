#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from collections import Counter

def main(cache_dir: Path, max_print: int):
    files = list(cache_dir.rglob("*.pkl"))
    print(f"[INFO] Found {len(files)} cache files")

    ok = 0
    missing = 0
    examples = []

    for p in files:
        try:
            with open(p, "rb") as f:
                feat = pickle.load(f)
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
            continue

        # PlutoFeature usually stores data here
        data = getattr(feat, "data", None)
        if data is None:
            missing += 1
            if len(examples) < max_print:
                examples.append((p, "NO_DATA"))
            continue

        emoe = data.get("emoe", None)
        if emoe is None or "emoe_class_id" not in emoe:
            missing += 1
            if len(examples) < max_print:
                tok = data.get("token") or data.get("scenario_token")
                examples.append((p, tok))
        else:
            ok += 1

    print("\n========== SUMMARY ==========")
    print("with emoe:", ok)
    print("missing emoe:", missing)
    print("total checked:", ok + missing)

    if examples:
        print("\nExamples missing emoe:")
        for p, tok in examples:
            print(" -", p, "token=", tok)

    if missing == 0:
        print("\n[OK] All cached features contain emoe")
    else:
        print("\n[ERROR] Cache is inconsistent: emoe missing")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Path to feature cache root")
    ap.add_argument("--max_print", type=int, default=10)
    args = ap.parse_args()

    main(Path(args.cache_dir).expanduser().resolve(), args.max_print)
