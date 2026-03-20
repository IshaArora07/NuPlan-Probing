#!/usr/bin/env python3
"""
Check for token collisions across nuPlan database files and between
the precompute labels and the feature cache.
"""

import os
import gzip
import json
import pickle
import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]

WRONG_SIDE = {
    0: lambda y: y < 0,
    2: lambda y: y > 0,
}


def load_emoe_class(feat_path):
    try:
        raw = pickle.load(gzip.open(feat_path, "rb"))
        inner = raw["data"]
        if hasattr(inner, "data"):
            inner = inner.data

        if not isinstance(inner, dict):
            return None

        emoe = inner.get("emoe")
        if emoe is None:
            return None

        val = emoe.get("emoe_class_id")
        if val is None:
            return None

        if hasattr(val, "item"):
            val = val.item()

        return int(val)

    except Exception:
        return None


def load_traj_endpoint(traj_path):
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))
        arr = np.array(raw["data"] if isinstance(raw, dict) else raw)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[-1, 0]), float(arr[-1, 1])

        return None

    except Exception:
        return None


def get_wrong_side_tokens(cache_dir, class_id):
    """Find tokens with wrong-side trajectory."""
    wrong_side_fn = WRONG_SIDE.get(class_id, lambda y: False)
    tokens = []

    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue

        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue

            for tok_dir in sorted(tag.iterdir()):
                if not tok_dir.is_dir():
                    continue

                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"

                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)
                if cid != class_id:
                    continue

                ep = load_traj_endpoint(traj_p)
                if ep is None:
                    continue

                if wrong_side_fn(ep[1]):
                    tokens.append({
                        "token": tok_dir.name,
                        "traj_x": ep[0],
                        "traj_y": ep[1],
                        "log_dir": log.name,
                    })

    return tokens


def query_token_in_db(db_path, token):
    results = []

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        for table in ["lidarpc", "ego_pose", "scene"]:
            try:
                cur.execute(
                    f"SELECT token, log_name FROM {table} WHERE token = ? LIMIT 5",
                    (token,),
                )
                rows = cur.fetchall()

                if rows:
                    for row in rows:
                        results.append({
                            "db": db_path.name,
                            "table": table,
                            "token": row[0],
                            "log_name": row[1] if len(row) > 1 else "?",
                        })
                    break

            except sqlite3.OperationalError:
                continue

        conn.close()

    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--class_id", type=int, default=0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    data_root = os.environ["NUPLAN_DATA_ROOT"]
    db_dir = Path(data_root) / "nuplan-v1.1" / "splits" / args.split
    db_files = sorted(db_dir.glob("*.db"))

    # load labels
    label_map = {}
    with open(args.labels_path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                label_map[r["token"]] = r
            except Exception:
                continue

    print(f"[INFO] Loaded {len(label_map)} labels")
    print(f"[INFO] Found {len(db_files)} db files")

    # wrong-side tokens
    wrong_tokens = get_wrong_side_tokens(cache_dir, args.class_id)

    print(f"\nFound {len(wrong_tokens)} wrong-side tokens")

    if not wrong_tokens:
        print("No issues found")
        return

    collision_found = False

    for t in wrong_tokens:
        tok = t["token"]

        print(f"\nToken: {tok}")

        # label info
        rec = label_map.get(tok)
        if rec:
            print(f"  Label class: {rec.get('emoe_class_id')}")
        else:
            print("  Not found in labels")

        # db search
        db_hits = []
        for db in db_files:
            db_hits.extend(query_token_in_db(db, tok))

        if len(db_hits) > 1:
            collision_found = True
            print("  COLLISION:")
            for h in db_hits:
                print(f"    {h['db']} → {h['log_name']}")
        elif len(db_hits) == 1:
            print(f"  Found in {db_hits[0]['db']}")
        else:
            print("  Not found in any db")

    # summary
    print("\n=== SUMMARY ===")

    if collision_found:
        print("✗ Token collisions detected")
        print("→ Precompute and cache may use different scenarios")
    else:
        print("✓ No token collisions found")


if __name__ == "__main__":
    main()
