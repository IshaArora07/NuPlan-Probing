#!/usr/bin/env python3
"""
Build a balanced 200k EMoE subset from multiple scene_labels.jsonl files,
compute KMeans anchors, and export:

- filtered scene labels JSONL
- anchors.npy
- scenarios.yaml
- summary.txt

Assumptions about each JSONL line:
{
  "token": "<scenario_token>",
  "emoe_class_id": <int in [0..5]>,
  "endpoint": [x, y]         # OPTIONAL: if absent, you must replace get_endpoint() logic
  ... (other fields are preserved)
}
"""

import argparse
import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for KMeans. Install with `pip install scikit-learn`."
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scene_labels(jsonl_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple *.jsonl files into a single list of records.

    If tokens appear multiple times across files, the *last* occurrence wins.
    """
    token_to_rec: Dict[str, Dict[str, Any]] = {}
    for path in jsonl_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                token = str(rec["token"])
                token_to_rec[token] = rec
    return list(token_to_rec.values())


def get_endpoint(rec: Dict[str, Any]) -> Tuple[float, float]:
    """
    Extract endpoint (x, y) from record.

    EXPECTED:
        rec["endpoint"] = [x, y]   OR
        rec["endpoint_x"], rec["endpoint_y"]

    If your JSONL has a different structure, change this function accordingly.
    """
    if "endpoint" in rec:
        x, y = rec["endpoint"]
        return float(x), float(y)
    elif "endpoint_x" in rec and "endpoint_y" in rec:
        return float(rec["endpoint_x"]), float(rec["endpoint_y"])
    else:
        raise KeyError(
            "Endpoint not found in record. "
            "Add [x, y] as 'endpoint' or 'endpoint_x'/'endpoint_y' or adapt get_endpoint()."
        )


def stratified_sample(
    records: List[Dict[str, Any]],
    total_target: int = 200_000,
    num_classes: int = 6,
    fixed_classes: List[int] = [4, 5],
    rng_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    - Use ALL records from fixed_classes (e.g. 4,5).
    - Fill remaining budget from other classes equally (as much as possible).

    Returns the selected records (shuffled).
    """
    rng = random.Random(rng_seed)

    # Group by class
    per_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cid = int(rec["emoe_class_id"])
        if cid < 0 or cid >= num_classes:
            raise ValueError(f"Invalid class id {cid} in record with token {rec.get('token')}")
        per_class[cid].append(rec)

    # Counts
    counts = {c: len(rs) for c, rs in per_class.items()}
    print("Original counts per class:", counts)

    # Take all fixed classes
    selected: List[Dict[str, Any]] = []
    fixed_count = 0
    for c in fixed_classes:
        cls_recs = per_class.get(c, [])
        selected.extend(cls_recs)
        fixed_count += len(cls_recs)

    if fixed_count > total_target:
        raise ValueError(
            f"Fixed classes {fixed_classes} already exceed target: "
            f"{fixed_count} > {total_target}"
        )

    remaining_target = total_target - fixed_count
    other_classes = [c for c in range(num_classes) if c not in fixed_classes]

    # distribute remaining equally across other_classes
    base = remaining_target // len(other_classes)
    leftover = remaining_target % len(other_classes)

    for i, c in enumerate(sorted(other_classes)):
        desired = base + (1 if i < leftover else 0)
        available = len(per_class.get(c, []))
        if available < desired:
            # Cap at available; you can implement redistribution if needed
            print(
                f"Warning: class {c} has only {available} samples, "
                f"desired {desired}. Using {available}."
            )
            desired = available

        cls_recs = per_class.get(c, [])
        rng.shuffle(cls_recs)
        selected.extend(cls_recs[:desired])

    rng.shuffle(selected)

    final_counts = Counter(int(r["emoe_class_id"]) for r in selected)
    print("Selected counts per class:", dict(final_counts))
    print("Total selected:", len(selected))

    return selected


def compute_kmeans_anchors(
    records: List[Dict[str, Any]],
    num_classes: int,
    n_anchors_per_class: int = 24,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Compute per-class KMeans anchors on endpoints.

    Returns:
        anchors: np.ndarray of shape [num_classes, n_anchors_per_class, 2]
    """
    anchors = np.zeros((num_classes, n_anchors_per_class, 2), dtype=np.float32)
    rng = np.random.RandomState(rng_seed)

    # group endpoints per class
    endpoints_per_class: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for rec in records:
        cid = int(rec["emoe_class_id"])
        x, y = get_endpoint(rec)
        endpoints_per_class[cid].append((x, y))

    for c in range(num_classes):
        pts = endpoints_per_class.get(c, [])
        if len(pts) == 0:
            print(f"Warning: no endpoints for class {c}; anchors will be zeros.")
            continue

        pts_arr = np.array(pts, dtype=np.float32)

        if len(pts_arr) < n_anchors_per_class:
            print(
                f"Warning: class {c} has only {len(pts_arr)} points; "
                f"KMeans(n_clusters={n_anchors_per_class}) may be unstable."
            )
            n_clusters = min(len(pts_arr), n_anchors_per_class)
        else:
            n_clusters = n_anchors_per_class

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=rng,
            n_init=10,
        )
        kmeans.fit(pts_arr)
        centers = kmeans.cluster_centers_  # [n_clusters, 2]

        anchors[c, :n_clusters] = centers
        # If n_clusters < n_anchors_per_class, remaining rows stay as zeros.

    return anchors


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_scenarios_yaml(path: str, records: List[Dict[str, Any]]) -> None:
    """
    Write a simple YAML file listing scenario tokens.

    Format:
        scenario_tokens:
          - "token1"
          - "token2"
          ...
    """
    tokens = [str(r["token"]) for r in records]
    with open(path, "w") as f:
        f.write("scenario_tokens:\n")
        for t in tokens:
            f.write(f'  - "{t}"\n')


def write_summary(
    path: str,
    original_records: List[Dict[str, Any]],
    selected_records: List[Dict[str, Any]],
    anchors: np.ndarray,
) -> None:
    orig_counts = Counter(int(r["emoe_class_id"]) for r in original_records)
    sel_counts = Counter(int(r["emoe_class_id"]) for r in selected_records)

    with open(path, "w") as f:
        f.write("=== EMoE Subset Summary ===\n\n")
        f.write("Original counts per class:\n")
        for c in sorted(orig_counts):
            f.write(f"  class {c}: {orig_counts[c]}\n")
        f.write("\nSelected counts per class:\n")
        for c in sorted(sel_counts):
            f.write(f"  class {c}: {sel_counts[c]}\n")
        f.write(f"\nTotal selected: {len(selected_records)}\n\n")

        f.write("Anchors shape: {}\n".format(list(anchors.shape)))
        f.write("Anchors per class: {}\n\n".format(anchors.shape[1]))

        # quick diagnostics: centroid norms per class
        norms = np.linalg.norm(anchors, axis=-1)  # [C, K]
        for c in range(anchors.shape[0]):
            nonzero = norms[c][norms[c] > 0]
            if len(nonzero) == 0:
                f.write(f"class {c}: no nonzero anchors\n")
            else:
                f.write(
                    f"class {c}: {len(nonzero)} anchors, "
                    f"mean |center|={nonzero.mean():.3f}, "
                    f"min={nonzero.min():.3f}, max={nonzero.max():.3f}\n"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 200k EMoE subset, compute anchors and export configs."
    )
    parser.add_argument(
        "--input_jsonl",
        nargs="+",
        required=True,
        help="List of scene_labels.jsonl files to merge.",
    )
    parser.add_argument(
        "--output_labels",
        type=str,
        default="scene_labels_200k.jsonl",
        help="Path to filtered scene labels JSONL.",
    )
    parser.add_argument(
        "--output_anchors",
        type=str,
        default="anchors.npy",
        help="Path to anchors .npy file.",
    )
    parser.add_argument(
        "--output_scenarios",
        type=str,
        default="scenarios.yaml",
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="summary.txt",
        help="Path to summary text file.",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=200_000,
        help="Total target number of samples.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of EMoE classes (0..num_classes-1).",
    )
    parser.add_argument(
        "--fixed_classes",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Classes to take all samples from (no downsampling).",
    )
    parser.add_argument(
        "--anchors_per_class",
        type=int,
        default=24,
        help="Number of KMeans anchors per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and KMeans.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_labels)) or ".", exist_ok=True)

    # 1) Load all labels
    original_records = load_scene_labels(args.input_jsonl)

    # 2) Stratified sampling to 200k with fixed classes 4,5
    selected_records = stratified_sample(
        original_records,
        total_target=args.total_samples,
        num_classes=args.num_classes,
        fixed_classes=args.fixed_classes,
        rng_seed=args.seed,
    )

    # 3) Compute KMeans anchors on endpoints for the selected subset
    anchors = compute_kmeans_anchors(
        selected_records,
        num_classes=args.num_classes,
        n_anchors_per_class=args.anchors_per_class,
        rng_seed=args.seed,
    )

    # 4) Write outputs
    write_jsonl(args.output_labels, selected_records)
    np.save(args.output_anchors, anchors)
    write_scenarios_yaml(args.output_scenarios, selected_records)
    write_summary(
        args.summary_path, original_records, selected_records, anchors
    )

    print("Done.")
    print(f"  Labels:    {args.output_labels}")
    print(f"  Anchors:   {args.output_anchors}")
    print(f"  Scenarios: {args.output_scenarios}")
    print(f"  Summary:   {args.summary_path}")


if __name__ == "__main__":
    main()
