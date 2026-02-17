#!/usr/bin/env python3
"""
Build EMoE scene anchors from trajectory endpoints.

Input:
  - scene_labels.jsonl (from 2-class classifier)
    Each line must contain:
      {
        "token": str,
        "emoe_class_id": int (0 or 1),
        "trajectory_endpoint_xy": [x, y]
      }

Output:
  - scene_anchors.npy           shape [2, Ka, 2]
  - scene_anchor_tokens.json    tokens used per class

Behavior:
  - Selects EXACTLY 2 scenarios per class
  - Runs KMeans per class
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
NUM_CLASSES = 2
SCENARIOS_PER_CLASS = 2


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Ka", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = out_dir / "scene_anchors.npy"
    tokens_path = out_dir / "scene_anchor_tokens.json"

    # ------------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------------
    per_class_points = defaultdict(list)
    per_class_tokens = defaultdict(list)

    with open(args.labels_jsonl, "r") as f:
        for line in f:
            rec = json.loads(line)
            cid = int(rec["emoe_class_id"])
            pt = np.asarray(rec["trajectory_endpoint_xy"], dtype=np.float32)
            tok = rec["token"]

            if cid in (0, 1):
                per_class_points[cid].append(pt)
                per_class_tokens[cid].append(tok)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    for c in range(NUM_CLASSES):
        if len(per_class_points[c]) < SCENARIOS_PER_CLASS:
            raise RuntimeError(
                f"Class {c} has only {len(per_class_points[c])} samples, "
                f"need {SCENARIOS_PER_CLASS}"
            )

    # ------------------------------------------------------------------
    # Build anchors
    # ------------------------------------------------------------------
    Ka = int(args.Ka)
    scene_anchors = np.zeros((NUM_CLASSES, Ka, 2), dtype=np.float32)
    used_tokens = {}

    rng = np.random.RandomState(args.seed)

    for c in range(NUM_CLASSES):
        idx = rng.choice(
            len(per_class_points[c]),
            size=SCENARIOS_PER_CLASS,
            replace=False,
        )

        pts = np.stack([per_class_points[c][i] for i in idx], axis=0)
        toks = [per_class_tokens[c][i] for i in idx]

        used_tokens[c] = toks

        # KMeans (with Ka <= num points)
        n_clusters = min(Ka, pts.shape[0])
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            n_init=10,
        )
        kmeans.fit(pts)

        centers = kmeans.cluster_centers_.astype(np.float32)

        scene_anchors[c, :n_clusters, :] = centers

        # If Ka > n_clusters, repeat first center
        if n_clusters < Ka:
            scene_anchors[c, n_clusters:, :] = centers[0:1]

        print(
            f"[INFO] class={c}  "
            f"used_tokens={toks}  "
            f"anchors={centers.tolist()}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.save(anchors_path, scene_anchors)

    with open(tokens_path, "w") as f:
        json.dump(used_tokens, f, indent=2)

    print(f"\n[INFO] Saved anchors to: {anchors_path}")
    print(f"[INFO] Saved anchor tokens to: {tokens_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
