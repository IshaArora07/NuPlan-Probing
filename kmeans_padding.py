#!/usr/bin/env python3
"""
Standalone EMoE anchor generation from scene_labels.jsonl.

Reads endpoint_xy and emoe_class_id from an existing scene_labels.jsonl,
runs improved KMeans per class (with perturbed padding instead of repeated
centers), and outputs:

- scene_anchors.npy  : shape [num_classes, Ka, 2]
- anchor_summary.json: per-class stats, cluster separation, endpoint counts

Instead of repeating centers[:1] to fill missing clusters,
this perturbs existing centers with small Gaussian noise so all Ka anchors
stay distinct.

Example:
python generate_anchors.py \
    --labels_path /path/to/scene_labels.jsonl \
    --output_dir /path/to/output \
    --Ka 32 \
    --num_classes 6
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]


def load_endpoints(
    labels_path: str,
    num_classes: int,
    min_travel_distance: float,
) -> Dict[int, List[np.ndarray]]:
    """
    Read scene_labels.jsonl and group endpoints by class.
    """
    endpoints_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)

    total_read = 0
    total_skipped_class = 0
    total_skipped_distance = 0
    total_skipped_missing = 0

    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_read += 1
            record = json.loads(line)

            cls = record.get("emoe_class_id", -1)
            if not (0 <= cls < num_classes):
                total_skipped_class += 1
                continue

            dist = record.get("travel_distance_m", 0.0)
            if dist < min_travel_distance:
                total_skipped_distance += 1
                continue

            ep = record.get("endpoint_xy", None)
            if ep is None or len(ep) != 2:
                total_skipped_missing += 1
                continue

            x, y = float(ep[0]), float(ep[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                total_skipped_missing += 1
                continue

            endpoints_by_class[cls].append(np.array([x, y], dtype=np.float32))

    print(f"[INFO] Records read: {total_read}")
    print(f"[INFO] Skipped (class): {total_skipped_class}")
    print(f"[INFO] Skipped (distance): {total_skipped_distance}")
    print(f"[INFO] Skipped (missing/nan): {total_skipped_missing}")
    print(f"[INFO] Usable endpoints: {sum(len(v) for v in endpoints_by_class.values())}")

    return endpoints_by_class


def min_pairwise_dist(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return float("inf")

    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(float(np.linalg.norm(centers[i] - centers[j])))

    return float(min(dists))


def mean_pairwise_dist(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 0.0

    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(float(np.linalg.norm(centers[i] - centers[j])))

    return float(np.mean(dists))


def run_kmeans_with_perturbed_padding(
    pts: np.ndarray,
    Ka: int,
    noise_std: float,
    kmeans_seed: int,
) -> np.ndarray:
    """
    Run KMeans and pad missing anchors with noisy copies.
    """
    n_pts = pts.shape[0]

    if n_pts == 1:
        centers = pts.copy()
    else:
        n_clusters = min(Ka, n_pts)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=kmeans_seed,
            n_init=10,   # cluster-safe version
            max_iter=500,
        )
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_.astype(np.float32)

    n_clusters = centers.shape[0]

    if n_clusters == Ka:
        return centers

    reps = Ka - n_clusters
    rng = np.random.RandomState(kmeans_seed + 1)

    source_idx = rng.choice(n_clusters, size=reps, replace=True)
    noise = rng.randn(reps, 2).astype(np.float32) * float(noise_std)
    extra = centers[source_idx] + noise

    return np.concatenate([centers, extra], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate EMoE scene anchors from scene_labels.jsonl."
    )
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Ka", type=int, default=24)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--min_travel_distance", type=float, default=5.0)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--kmeans_seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = out_dir / "scene_anchors.npy"
    summary_path = out_dir / "anchor_summary.json"

    Ka = int(args.Ka)
    num_classes = int(args.num_classes)

    print(f"[INFO] labels_path = {args.labels_path}")
    print(f"[INFO] output_dir = {out_dir}")
    print(f"[INFO] Ka = {Ka}")
    print(f"[INFO] num_classes = {num_classes}")
    print()

    endpoints_by_class = load_endpoints(
        labels_path=args.labels_path,
        num_classes=num_classes,
        min_travel_distance=args.min_travel_distance,
    )

    scene_anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)
    summary = {
        "Ka": Ka,
        "num_classes": num_classes,
        "classes": {},
    }

    print("[INFO] Running KMeans per class...")

    for c in range(num_classes):
        class_name = EMOE_SCENE_TYPES[c]
        pts = np.asarray(endpoints_by_class.get(c, []), dtype=np.float32)
        n_pts = pts.shape[0]

        if n_pts == 0:
            print(f"[WARN] class {c} ({class_name}): no endpoints")
            continue

        centers = run_kmeans_with_perturbed_padding(
            pts=pts,
            Ka=Ka,
            noise_std=args.noise_std,
            kmeans_seed=args.kmeans_seed,
        )

        scene_anchors[c] = centers

        min_sep = min_pairwise_dist(centers)
        mean_sep = mean_pairwise_dist(centers)

        print(
            f"class {c} ({class_name:28s}): "
            f"{n_pts:7d} pts | min_sep={min_sep:.2f}m mean_sep={mean_sep:.2f}m"
        )

        summary["classes"][str(c)] = {
            "class_name": class_name,
            "n_endpoints": int(n_pts),
            "min_pairwise_dist_m": round(min_sep, 3),
            "mean_pairwise_dist_m": round(mean_sep, 3),
        }

    np.save(anchors_path, scene_anchors)
    print(f"\n[INFO] Saved anchors: {anchors_path}")
    print(f"[INFO] Shape: {scene_anchors.shape}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved summary: {summary_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
