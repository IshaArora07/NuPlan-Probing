import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
]


def load_endpoints(
    labels_path: str,
    num_classes: int,
    min_travel_distance: float,
    x_cap: float,
    y_cap: float,
) -> Dict[int, List[np.ndarray]]:
    endpoints_by_class = defaultdict(list)

    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            cls = record.get("emoe_class_id", -1)
            if not (0 <= cls < num_classes):
                continue

            dist = record.get("travel_distance_m", 0.0)
            if dist < min_travel_distance:
                continue

            ep = record.get("endpoint_xy", None)
            if ep is None or len(ep) != 2:
                continue

            x, y = float(ep[0]), float(ep[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            x = np.clip(x, -x_cap, x_cap)
            y = np.clip(y, -y_cap, y_cap)

            endpoints_by_class[cls].append(
                np.array([x, y], dtype=np.float32)
            )

    return endpoints_by_class


def get_mandatory_anchors(c: int) -> np.ndarray:
    mandatory = []

    if c == 0:  # LEFT TURN
        # stop / wait before entering
        mandatory += [
            [2.0, 0.0],
            [5.0, 0.0],
        ]
        # cautious left entry
        mandatory += [
            [8.0, 2.0],
            [10.0, 4.0],
        ]
        # medium turn arc
        mandatory += [
            [12.0, 6.0],
            [14.0, 8.0],
        ]
        # deeper left completion
        mandatory += [
            [16.0, 10.0],
            [18.0, 12.0],
        ]

    elif c == 1:  # STRAIGHT INTERSECTION
        mandatory += [
            [2.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
            [12.0, 0.0],
        ]
        # multi-scale lateral avoidance
        mandatory += [
            [12.0, -2.0],
            [12.0,  2.0],
            [18.0, -3.0],
            [18.0,  3.0],
            [24.0, -3.7],
            [24.0,  3.7],
            [30.0, -3.7],
            [30.0,  3.7],
        ]

    elif c == 2:  # RIGHT TURN
        mandatory += [
            [2.0, 0.0],
            [5.0, 0.0],
        ]
        # cautious right entry
        mandatory += [
            [8.0, -2.0],
            [10.0, -4.0],
        ]
        # medium arc
        mandatory += [
            [12.0, -6.0],
            [14.0, -8.0],
        ]
        # wider comfort arc
        mandatory += [
            [16.0, -9.0],
            [18.0, -10.0],
        ]

    elif c == 3:  # STRAIGHT NON-INTERSECTION
        mandatory += [
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
            [8.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
        ]
        # richer lateral + lane-change geometry
        mandatory += [
            [12.0, -2.0],
            [12.0,  2.0],
            [18.0, -3.0],
            [18.0,  3.0],
            [24.0, -3.7],
            [24.0,  3.7],
            [30.0, -3.7],
            [30.0,  3.7],
        ]

    elif c == 4:  # ROUNDABOUT
        mandatory += [
            [8.0, 3.0],
            [12.0, 6.0],
            [16.0, 8.0],
        ]

    elif c == 5:  # U-TURN
        mandatory += [
            [4.0, 2.0],
            [6.0, 5.0],
            [8.0, 8.0],
            [6.0, 10.0],
        ]

    return np.array(mandatory, dtype=np.float32)


def run_kmeans_with_padding(
    pts: np.ndarray,
    n_clusters: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    if len(pts) == 0:
        rng = np.random.RandomState(seed)
        return rng.randn(n_clusters, 2).astype(np.float32)

    k = min(n_clusters, len(pts))
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=500,
    )
    km.fit(pts)
    centers = km.cluster_centers_.astype(np.float32)

    if len(centers) < n_clusters:
        rng = np.random.RandomState(seed + 1)
        needed = n_clusters - len(centers)
        idx = rng.choice(len(centers), size=needed, replace=True)
        noise = rng.randn(needed, 2).astype(np.float32) * noise_std
        extra = centers[idx] + noise
        centers = np.concatenate([centers, extra], axis=0)

    return centers[:n_clusters]


def generate_anchors(
    labels_path: str,
    output_path: str,
    Ka: int = 24,
    num_classes: int = 6,
    x_cap: float = 35.0,
    y_cap: float = 20.0,
    min_travel: float = 1.0,
    noise_std: float = 0.75,
    seed: int = 42,
):
    endpoints_by_class = load_endpoints(
        labels_path,
        num_classes,
        min_travel,
        x_cap,
        y_cap,
    )

    scene_anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)

    for c in range(num_classes):
        pts = np.array(endpoints_by_class.get(c, []), dtype=np.float32)

        mandatory = get_mandatory_anchors(c)
        n_mand = len(mandatory)
        n_kmeans = Ka - n_mand

        kmeans = run_kmeans_with_padding(
            pts,
            n_kmeans,
            noise_std,
            seed + c,
        )

        anchors = np.concatenate([mandatory, kmeans], axis=0)
        scene_anchors[c] = anchors[:Ka]

        xs = scene_anchors[c, :, 0]
        ys = scene_anchors[c, :, 1]

        print(f"\nClass {c} {EMOE_SCENE_TYPES[c]}")
        print(f"x range: {xs.min():.2f} -> {xs.max():.2f}")
        print(f"y range: {ys.min():.2f} -> {ys.max():.2f}")
        print(f"mean |y|: {np.abs(ys).mean():.2f}")
        print(f"x<8: {(xs < 8).sum()}")
        print(f"8<=x<16: {((xs >= 8) & (xs < 16)).sum()}")
        print(f"x>=16: {(xs >= 16).sum()}")

    np.save(output_path, scene_anchors)
    print(f"\nSaved to {output_path} | shape={scene_anchors.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    generate_anchors(
        labels_path=args.labels_path,
        output_path=args.output_path,
    )
