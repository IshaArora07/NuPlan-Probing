#!/usr/bin/env python3
"""
EMoE 2-class classifier: STRAIGHT vs TURN

Goal:
- Make router supervision maximally learnable
- Enable clean overfitting on 1 batch
- Scale robustly to 1.2M scenarios

Classes:
  0: straight
  1: turn

Routing decision uses ONLY motion primitives:
- net heading change
- total curvature
- travel distance

All map semantics are removed from routing logic.
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


# ======================================================================================
# EMoE classes (2)
# ======================================================================================
EMOE_SCENE_TYPES = [
    "straight",  # 0
    "turn",      # 1
]


# ======================================================================================
# Motion thresholds (router-visible primitives)
# ======================================================================================
TURN_NET_HEADING_MIN = math.radians(25.0)      # |Δθ|
TURN_TOTAL_CURVATURE_MIN = math.radians(60.0) # Σ|Δθ|
MIN_TRAVEL_DIST = 1.0                          # meters


# ======================================================================================
# Helpers
# ======================================================================================
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, hs = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
        hs.append(float(ego.rear_axle.heading))
    return np.asarray(xs), np.asarray(ys), np.asarray(hs)


def path_length(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


def ego_endpoint_in_ego_frame(xs, ys, hs):
    if len(xs) < 2:
        return np.zeros(2, dtype=np.float32)

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    h0 = hs[0]

    c, s = math.cos(-h0), math.sin(-h0)
    return np.array([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)


# ======================================================================================
# Core classifier (THIS is what matters)
# ======================================================================================
def classify_turn_vs_straight(scenario) -> Tuple[int, str, Dict[str, Any]]:
    xs, ys, hs = compute_ego_xyh(scenario)

    if len(xs) < 3:
        return 0, "short_sequence", {}

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    dist = float(math.hypot(dx, dy))

    h0 = hs[0]
    hT = hs[-1]
    delta_heading = wrap_to_pi(hT - h0)
    abs_dh = abs(delta_heading)

    dh = np.diff(hs)
    dh = np.vectorize(wrap_to_pi)(dh)
    total_abs = float(np.sum(np.abs(dh)))

    debug = {
        "travel_distance_m": dist,
        "delta_heading_deg": math.degrees(delta_heading),
        "abs_delta_heading_deg": math.degrees(abs_dh),
        "total_abs_heading_deg": math.degrees(total_abs),
    }

    # ---- TURN decision ----
    if (
        dist >= MIN_TRAVEL_DIST
        and (abs_dh >= TURN_NET_HEADING_MIN or total_abs >= TURN_TOTAL_CURVATURE_MIN)
    ):
        return 1, "motion_turn", debug

    # ---- STRAIGHT fallback ----
    return 0, "motion_straight", debug


# ======================================================================================
# nuPlan scenario loading
# ======================================================================================
def build_scenarios(split: str, max_scenarios: int, num_workers: int):
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=num_workers)

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        limit_total_scenarios=max_scenarios if max_scenarios > 0 else None,
    )

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=num_workers,
    )

    return builder.get_scenarios(scenario_filter, worker)


# ======================================================================================
# Main
# ======================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Ka", type=int, default=24)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "scene_labels.jsonl"
    anchors_path = out_dir / "scene_anchors.npy"

    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)
    print(f"[INFO] Loaded {len(scenarios)} scenarios")

    endpoints_by_class = defaultdict(list)
    class_counts = Counter()

    with labels_path.open("w") as f:
        for scenario in tqdm(scenarios):
            cls_id, stage, debug = classify_turn_vs_straight(scenario)
            class_counts[cls_id] += 1

            xs, ys, hs = compute_ego_xyh(scenario)
            if path_length(xs, ys) >= MIN_TRAVEL_DIST:
                endpoints_by_class[cls_id].append(
                    ego_endpoint_in_ego_frame(xs, ys, hs)
                )

            record = {
                "token": scenario.token,
                "emoe_class_id": cls_id,
                "emoe_class_name": EMOE_SCENE_TYPES[cls_id],
                "stage": stage,
                "debug": debug,
            }
            f.write(json.dumps(record) + "\n")

    print("[INFO] Class distribution:")
    for k, v in class_counts.items():
        print(f"  {EMOE_SCENE_TYPES[k]:10s}: {v}")

    # ---- Anchors ----
    scene_anchors = np.zeros((2, args.Ka, 2), dtype=np.float32)

    for c in range(2):
        pts = np.asarray(endpoints_by_class[c], dtype=np.float32)
        if len(pts) == 0:
            continue

        k = min(args.Ka, len(pts))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(pts)

        scene_anchors[c, :k] = kmeans.cluster_centers_
        if k < args.Ka:
            scene_anchors[c, k:] = scene_anchors[c, 0]

    np.save(anchors_path, scene_anchors)
    print(f"[INFO] Saved labels → {labels_path}")
    print(f"[INFO] Saved anchors → {anchors_path}")


if __name__ == "__main__":
    main()
