#!/usr/bin/env python3
"""
Concatenate multiple classification JSON/JSONL outputs and generate:
1) scene_labels.jsonl  (token -> emoe_class_id)
2) G_scene_anchors.npy (scene-type anchors) with shape [C, Ka, 2] where C=6
3) meta json with counts etc.

Assumptions (per your message):
- No duplicates across files.
- Each record contains:
    - token (scenario.token)
    - emoe_class_id (0..5)
    - endpoint_xy (ego-frame endpoint [x, y])
- C = 6 classes (no roundabout)
- Ka = 24
"""

import argparse
import glob
import json
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    """
    Yields dict records from:
      - JSONL file (1 json per line), or
      - JSON file containing a list of dicts.
    """
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path}: JSON must be a list if it starts with '['")
            for r in data:
                if isinstance(r, dict):
                    yield r
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if isinstance(r, dict):
                    yield r


def kmeans_centers(points: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """
    Dependency-free k-means on 2D points.
      points: [N, 2]
      returns centers: [k, 2]
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if N < k:
        raise RuntimeError(f"Need at least N>=k points, got N={N}, k={k}")

    centers = points[rng.choice(N, size=k, replace=False)]
    for _ in range(iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [N,k]
        idx = d2.argmin(axis=1)  # [N]

        new_centers = centers.copy()
        for j in range(k):
            mask = (idx == j)
            if np.any(mask):
                new_centers[j] = points[mask].mean(axis=0)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help='Glob for your per-db outputs, e.g. "out/*.jsonl" or "out/*.json"',
    )
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--Ka", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token_key", type=str, default="token")
    ap.add_argument("--class_key", type=str, default="emoe_class_id")
    ap.add_argument("--endpoint_key", type=str, default="endpoint_xy")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched glob: {args.input_glob}")

    # Collect records
    labels_out_path = f"{args.out_dir.rstrip('/')}/scene_labels.jsonl"
    anchors_out_path = f"{args.out_dir.rstrip('/')}/G_scene_anchors.npy"
    meta_out_path = f"{args.out_dir.rstrip('/')}/G_scene_anchors_meta.json"

    # Buckets of endpoints per class
    endpoints: List[List[List[float]]] = [[] for _ in range(args.num_classes)]

    num_rows = 0
    num_written_labels = 0
    bad_rows = 0

    with open(labels_out_path, "w") as labels_f:
        for p in paths:
            for r in iter_json_records(p):
                num_rows += 1

                if args.token_key not in r or args.class_key not in r or args.endpoint_key not in r:
                    bad_rows += 1
                    continue

                token = r[args.token_key]
                class_id = int(r[args.class_key])
                ep = r[args.endpoint_key]

                if not isinstance(token, str) or not token:
                    bad_rows += 1
                    continue

                if class_id < 0 or class_id >= args.num_classes:
                    bad_rows += 1
                    continue

                if not (isinstance(ep, (list, tuple)) and len(ep) == 2):
                    bad_rows += 1
                    continue

                x, y = float(ep[0]), float(ep[1])
                endpoints[class_id].append([x, y])

                # Write labels jsonl (token-keyed)
                labels_f.write(json.dumps({"token": token, "emoe_class_id": class_id}) + "\n")
                num_written_labels += 1

    counts = np.array([len(b) for b in endpoints], dtype=np.int64)
    if np.any(counts < args.Ka):
        bad = np.where(counts < args.Ka)[0].tolist()
        raise SystemExit(
            f"Not enough samples for Ka={args.Ka} in class IDs {bad}. "
            f"Counts per class: {counts.tolist()}"
        )

    # Compute anchors: G[class_id] = centers [Ka,2]
    G = np.zeros((args.num_classes, args.Ka, 2), dtype=np.float32)
    for c in range(args.num_classes):
        pts = np.asarray(endpoints[c], dtype=np.float32)  # [N,2]
        G[c] = kmeans_centers(pts, args.Ka, seed=args.seed + c)

    np.save(anchors_out_path, G)

    meta = {
        "num_classes": args.num_classes,
        "Ka": args.Ka,
        "counts_per_class": counts.tolist(),
        "num_input_files": len(paths),
        "num_rows_read": num_rows,
        "num_labels_written": num_written_labels,
        "bad_rows_skipped": bad_rows,
        "frame": "ego@t0 (as provided in endpoint_xy)",
        "keys": {"token": args.token_key, "class": args.class_key, "endpoint": args.endpoint_key},
        "outputs": {
            "scene_labels_jsonl": labels_out_path,
            "G_scene_anchors_npy": anchors_out_path,
        },
    }

    with open(meta_out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", labels_out_path)
    print("Saved:", anchors_out_path, "shape=", G.shape)
    print("Meta:", meta_out_path)
    print("Counts per class:", counts.tolist())


if __name__ == "__main__":
    main()
