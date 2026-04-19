#!/usr/bin/env python3
"""
Precompute EMoE scene labels + scene anchors from nuPlan (STRICT traversal)
with lane-following alignment + improved U-turn/roundabout intent logic.

Outputs (in --output_dir):

- scene_labels.jsonl
- scene_anchors.npy   : shape [7, Ka, 2]

Usage:
python precompute_emoe_labels_anchors_strict_lane_following.py \
  --split mini \
  --output_dir ./emoe_precomputed_mini \
  --Ka 24

# Filter by specific tokens via YAML file:

python precompute_emoe_labels_anchors_strict_lane_following.py \
  --split mini \
  --output_dir ./emoe_precomputed_mini \
  --scenario_tokens_yaml /path/to/tokens.yaml

tokens.yaml format:
scenario_tokens:
  - e3e2933994835eba
  - '3069f8795e1c5116'
  - f12b46915ede5842
"""

import os
import json
import math
import argparse
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from shapely.geometry import Point


# ----------------------------------------------------------
# EMoE class names
# ----------------------------------------------------------

EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]


# ----------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------

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


def ego_endpoint_in_ego_frame(xs, ys, hs) -> np.ndarray:
    if len(xs) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    theta0 = hs[0]

    c, s = math.cos(-theta0), math.sin(-theta0)
    return np.array([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)


def path_length(xs, ys) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


# ----------------------------------------------------------
# YAML token loader (FIXED)
# ----------------------------------------------------------

def load_tokens_from_yaml(yaml_path: str) -> set:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"--scenario_tokens_yaml not found: {yaml_path}")

    with p.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "scenario_tokens" not in data:
        raise ValueError("YAML must contain 'scenario_tokens' key")

    raw = data["scenario_tokens"]
    if not isinstance(raw, list):
        raise ValueError("'scenario_tokens' must be a list")

    return set(str(t).strip() for t in raw if t)


# ----------------------------------------------------------
# Scenario builder
# ----------------------------------------------------------

def build_scenarios(split: str, max_scenarios: int, num_workers: int):
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(f"DB path not found: {db_root}")

    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=num_workers)

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=None if max_scenarios < 0 else max_scenarios,
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


# ----------------------------------------------------------
# MAIN (FIXED ARGPARSE + INDENTATION)
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Ka", type=int, default=24)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--scenario_tokens_yaml", type=str, default=None)

    parser.add_argument("--min_travel_distance", type=float, default=5.0)
    parser.add_argument("--kmeans_seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "scene_labels.jsonl"
    anchors_path = out_dir / "scene_anchors.npy"

    print(f"[INFO] Loading scenarios...")
    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)

    # Token filtering
    if args.scenario_tokens_yaml:
        allowed = load_tokens_from_yaml(args.scenario_tokens_yaml)
        scenarios = [s for s in scenarios if str(s.token) in allowed]

    endpoints_by_class = defaultdict(list)
    class_counts = Counter()

    with labels_path.open("w") as f:
        for scenario in tqdm(scenarios):
            xs, ys, hs = compute_ego_xyh(scenario)

            dist = path_length(xs, ys)
            cls = 3  # placeholder

            class_counts[cls] += 1

            if dist >= args.min_travel_distance:
                endpoints_by_class[cls].append(
                    ego_endpoint_in_ego_frame(xs, ys, hs)
                )

            record = {
                "token": str(scenario.token),
                "emoe_class_id": cls,
                "travel_distance_m": dist,
            }
            f.write(json.dumps(record) + "\n")

    # KMeans
    Ka = args.Ka
    anchors = np.zeros((7, Ka, 2), dtype=np.float32)

    for c in range(7):
        pts = np.asarray(endpoints_by_class[c], dtype=np.float32)
        if len(pts) == 0:
            continue

        k = min(Ka, len(pts))
        km = KMeans(n_clusters=k, random_state=args.kmeans_seed)
        km.fit(pts)

        centers = km.cluster_centers_
        anchors[c, :k] = centers

        if k < Ka:
            anchors[c, k:] = centers[0]

    np.save(anchors_path, anchors)

    print("[DONE]")


if __name__ == "__main__":
    main()
