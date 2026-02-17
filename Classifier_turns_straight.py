#!/usr/bin/env python3
"""
Precompute EMoE scene labels for 2-class routing:
  0 = straight
  1 = turn

Key goals:
- Router-friendly (motion-only semantics)
- Expert-specializable
- Scales cleanly to 1.2M scenarios
- No anchors computed here
- Trajectory endpoints saved for later KMeans

Outputs:
  - scene_labels.jsonl
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


# --------------------------------------------------------------------------------------
# EMoE classes (2 only)
# --------------------------------------------------------------------------------------
EMOE_SCENE_TYPES = [
    "straight",  # 0
    "turn",      # 1
]


# --------------------------------------------------------------------------------------
# Thresholds (router-visible motion primitives)
# --------------------------------------------------------------------------------------
TURN_NET_HEADING_MIN = math.radians(25.0)
TURN_TOTAL_HEADING_MIN = math.radians(60.0)
MIN_TRAJ_LEN = 3


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, hs = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
        hs.append(float(ego.rear_axle.heading))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(hs, dtype=np.float64),
    )


def ego_endpoint_in_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> np.ndarray:
    if len(xs) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    h0 = hs[0]

    c, s = math.cos(-h0), math.sin(-h0)
    return np.array([c * dx - s * dy, s * dx + c * dy], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Core 2-class classifier
# --------------------------------------------------------------------------------------
def classify_turn_vs_straight(
    xs: np.ndarray,
    ys: np.ndarray,
    hs: np.ndarray,
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns:
      emoe_class_id (0=straight, 1=turn)
      debug dict
    """
    T = len(xs)
    if T < MIN_TRAJ_LEN:
        return 0, {"reason": "short_trajectory"}

    h0, hT = float(hs[0]), float(hs[-1])
    delta_heading = wrap_to_pi(hT - h0)
    abs_delta_heading = abs(delta_heading)

    dh = np.diff(hs)
    dh = np.vectorize(wrap_to_pi)(dh)
    total_abs_heading = float(np.sum(np.abs(dh)))

    is_turn = (
        abs_delta_heading >= TURN_NET_HEADING_MIN
        or total_abs_heading >= TURN_TOTAL_HEADING_MIN
    )

    debug = {
        "delta_heading_deg": math.degrees(delta_heading),
        "abs_delta_heading_deg": math.degrees(abs_delta_heading),
        "total_abs_heading_deg": math.degrees(total_abs_heading),
        "turn_net_threshold_deg": math.degrees(TURN_NET_HEADING_MIN),
        "turn_total_threshold_deg": math.degrees(TURN_TOTAL_HEADING_MIN),
    }

    return (1 if is_turn else 0), debug


# --------------------------------------------------------------------------------------
# Scenario loading
# --------------------------------------------------------------------------------------
def build_scenarios(split: str, max_scenarios: int, num_workers: int) -> List[Any]:
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(db_root)

    worker = SingleMachineParallelExecutor(
        use_process_pool=False,
        num_workers=num_workers,
    )

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


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "scene_labels.jsonl"

    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)

    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    print(f"[INFO] Loaded {len(scenarios)} scenarios")

    class_counts = {0: 0, 1: 0}

    with labels_path.open("w") as f:
        for scenario in tqdm(scenarios, desc="Classifying"):
            xs, ys, hs = compute_ego_xyh(scenario)

            emoe_id, debug = classify_turn_vs_straight(xs, ys, hs)
            class_counts[emoe_id] += 1

            endpoint = ego_endpoint_in_ego_frame(xs, ys, hs)

            record = {
                "token": str(scenario.token),
                "emoe_class_id": int(emoe_id),
                "emoe_class_name": EMOE_SCENE_TYPES[emoe_id],
                "trajectory_endpoint_xy": endpoint.tolist(),
                "debug": debug,
            }

            f.write(json.dumps(record) + "\n")

    print("\n[INFO] Class distribution:")
    for k, v in class_counts.items():
        print(f"  {EMOE_SCENE_TYPES[k]:8s}: {v}")

    print(f"\n[INFO] Saved labels to {labels_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
