#!/usr/bin/env python3
"""
Precompute EMoE scene labels + scene anchors from nuPlan.

Outputs (in --output_dir):
  - scene_labels.jsonl  : one line per scenario (token -> emoe_class_id + debug + stage_used)
  - scene_anchors.npy   : shape [7, Ka, 2] (KMeans on GT trajectory endpoints per class)

Pipeline (stages):
  Stage 1 (tags/strings first):
    - roundabout -> class 4
    - u-turn     -> class 5
    - starting_right_turn / right_turn -> class 2   (DIRECT, as you requested)
    - other “turn-ish” tags -> sent to geometry/map verification (not directly left/right)

  Stage 2 (map semantics):
    - detect roundabout via map lane/roadblock properties if available
    - detect intersection presence
    - detect lane connector turn direction if available (LEFT/RIGHT/UTURN)

  Stage 3 (geometry, with FIXED logic):
    - HARD motion gate: stationary/slow-jitter can’t become LEFT/RIGHT
    - If intersection present (map OR tag hint):
        use NET heading (delta_heading) to decide LEFT/RIGHT/STRAIGHT at intersection
    - Outside intersection:
        only STRAIGHT_NON_INTERSECTION vs OTHERS (no left/right from curvature)

Notes on GPU:
  - This script is CPU-bound (nuPlan DB + map I/O + GeoPandas/Shapely internals).
  - Geometry math is cheap; GPU won’t speed up the dominant bottlenecks.

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python precompute_emoe_labels_anchors.py \
    --split mini \
    --output_dir ./emoe_precomputed_mini \
    --Ka 24 \
    --max_scenarios -1
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


# --------------------------------------------------------------------------------------
# EMoE class names
# --------------------------------------------------------------------------------------
EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
    "others",                         # 6
]


# --------------------------------------------------------------------------------------
# Geometry thresholds / gates (tuned to avoid stationary false turns)
# --------------------------------------------------------------------------------------
MIN_DIST_ANY = 5.0                         # ignore tiny motion entirely
MIN_DIST_TURN_AT_INTERSECTION = 12.0       # require real motion to call LEFT/RIGHT at intersection

NET_STRAIGHT_MAX_AT_INTERSECTION = math.radians(12.0)   # |Δθ| <= 12° -> straight at intersection
NET_TURN_MIN_AT_INTERSECTION = math.radians(18.0)       # |Δθ| >= 18° -> turn candidate
NET_TURN_MAX_AT_INTERSECTION = math.radians(165.0)      # avoid near-UTURN confusion

UTURN_CENTER = math.pi
UTURN_MARGIN = math.radians(35.0)

ROUNDABOUT_TOTAL_MIN = math.radians(270.0)              # fallback only
ROUNDABOUT_NET_MAX = math.radians(90.0)                 # fallback only


# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ego rear-axle x, y, heading over the scenario horizon."""
    xs, ys, hs = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(ego.rear_axle.x)
        ys.append(ego.rear_axle.y)
        hs.append(float(ego.rear_axle.heading))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(hs, dtype=np.float64)


def ego_endpoint_in_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Final endpoint in the initial ego frame: (x_rel, y_rel)."""
    if len(xs) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    x0, y0 = float(xs[0]), float(ys[0])
    xT, yT = float(xs[-1]), float(ys[-1])
    dx, dy = xT - x0, yT - y0

    theta0 = float(hs[0])
    c = math.cos(-theta0)
    s = math.sin(-theta0)
    x_rel = c * dx - s * dy
    y_rel = s * dx + c * dy
    return np.array([x_rel, y_rel], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Stage 1: tag/string-based priority mapping
# --------------------------------------------------------------------------------------
def _upper(x: Any) -> str:
    return str(x).upper() if x is not None else ""


def get_scenario_tags_if_available(scenario) -> List[str]:
    """
    Best-effort tag extraction. nuPlan devkit versions differ.
    If your scenario object exposes tags, we use them; otherwise empty list.
    """
    tags: List[str] = []
    # Common patterns people sometimes have in their own wrappers:
    for attr in ["tags", "scenario_tags", "log_tags"]:
        if hasattr(scenario, attr):
            try:
                val = getattr(scenario, attr)
                if isinstance(val, (list, tuple)):
                    tags.extend([_upper(t) for t in val])
            except Exception:
                pass

    # Some devkit versions expose scenario metadata methods; keep best-effort
    for method in ["get_tags", "get_scenario_tags"]:
        if hasattr(scenario, method):
            try:
                val = getattr(scenario, method)()
                if isinstance(val, (list, tuple)):
                    tags.extend([_upper(t) for t in val])
            except Exception:
                pass

    # de-dup preserving order
    seen = set()
    out = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def stage1_from_tags_and_type(scenario_type: str, tags: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (class_id, stage_name) if decided, else (None, None).

    Your requirement:
      - classify right turn tags directly as RIGHT
      - classify U-turn and roundabout directly
      - other turn-ish tags go to geometry verification (not direct left/right)
    """
    st = _upper(scenario_type)
    tset = set(tags)

    # Direct: roundabout
    if "ROUNDABOUT" in st or any("ROUNDABOUT" in t for t in tset):
        return 4, "stage1_tags"

    # Direct: u-turn
    if "UTURN" in st or "U_TURN" in st or any(("UTURN" in t or "U_TURN" in t) for t in tset):
        return 5, "stage1_tags"

    # Direct: right turn
    # (covers strings you mentioned like STARTING_RIGHT_TURN)
    if ("STARTING_RIGHT" in st and "TURN" in st) or ("RIGHT_TURN" in st) or any(("STARTING_RIGHT" in t and "TURN" in t) for t in tset):
        return 2, "stage1_tags"

    # Everything else: do NOT directly force left/right here (send to stage2/3)
    return None, None


def tag_intersection_hint(scenario_type: str, tags: List[str]) -> bool:
    """
    Conservative: treat these as intersection context hints.
    """
    st = _upper(scenario_type)
    tset = set(tags)
    keys = ["INTERSECTION", "TRAFFIC_LIGHT", "STOP_SIGN", "TRAVERSING_INTERSECTION", "ON_INTERSECTION"]
    if any(k in st for k in keys):
        return True
    if any(any(k in t for k in keys) for t in tset):
        return True
    return False


# --------------------------------------------------------------------------------------
# Stage 2: map-based flags (best-effort, version-robust)
# --------------------------------------------------------------------------------------
def get_lane_objects_along_trajectory(scenario, xs: np.ndarray, ys: np.ndarray, step: int = 5) -> List[Any]:
    """
    Sample along ego trajectory and retrieve lane or lane_connector objects via map_api.
    We subsample every 'step' frames to limit calls.
    """
    lanes: List[Any] = []
    map_api = getattr(scenario, "map_api", None)
    if map_api is None:
        return lanes

    for i in range(0, len(xs), step):
        x = float(xs[i])
        y = float(ys[i])
        lane = None
        try:
            if hasattr(map_api, "get_one_lane_or_lane_connector_from_point"):
                lane = map_api.get_one_lane_or_lane_connector_from_point(x, y)
            elif hasattr(map_api, "get_one_lane_or_lane_connector"):
                lane = map_api.get_one_lane_or_lane_connector(x, y)
        except Exception:
            lane = None
        if lane is not None:
            lanes.append(lane)
    return lanes


def map_based_flags_from_lanes(lanes: List[Any]) -> Tuple[bool, bool, Optional[int]]:
    """
    From lane/lane_connector objects, derive:
      - has_intersection
      - has_roundabout
      - lane_turn_class: Optional[int] in {0,2,5} if map strongly says left/right/uturn
    """
    has_intersection = False
    has_roundabout = False
    lane_turn_class: Optional[int] = None

    for lane in lanes:
        # Intersection / junction flags (varies by devkit)
        if bool(getattr(lane, "is_intersection", False)) or bool(getattr(lane, "in_junction", False)):
            has_intersection = True

        # Roadblock / lane group roundabout flags if present
        rb = getattr(lane, "roadblock", None)
        if rb is not None and bool(getattr(rb, "is_roundabout", False)):
            has_roundabout = True

        # Some versions have lane_group metadata
        lane_group = getattr(lane, "lane_group", None)
        if lane_group is not None:
            lg_type = str(getattr(lane_group, "type", "")).lower()
            if "roundabout" in lg_type:
                has_roundabout = True

        # Turn direction / connector type if present
        # (Important: we only use it if it is explicit)
        turn_dir = str(getattr(lane, "turn_direction", "")).lower()
        if not turn_dir:
            # some versions use turn_type or lane_connector_type
            turn_dir = str(getattr(lane, "turn_type", "")).lower()
        if not turn_dir:
            turn_dir = str(getattr(lane, "lane_connector_type", "")).lower()

        if "left" in turn_dir:
            lane_turn_class = 0
        elif "right" in turn_dir:
            lane_turn_class = 2
        elif "uturn" in turn_dir or "u_turn" in turn_dir or "u-turn" in turn_dir:
            lane_turn_class = 5

    return has_intersection, has_roundabout, lane_turn_class


def map_based_scene_flags(scenario, xs: np.ndarray, ys: np.ndarray, step: int = 5) -> Tuple[bool, bool, Optional[int]]:
    lanes = get_lane_objects_along_trajectory(scenario, xs, ys, step=step)
    if not lanes:
        return False, False, None
    return map_based_flags_from_lanes(lanes)


# --------------------------------------------------------------------------------------
# Stage 3: geometry + map fallback (FIXED)
# --------------------------------------------------------------------------------------
def classify_emoe_from_map_and_geometry(
    xs: np.ndarray,
    ys: np.ndarray,
    headings: np.ndarray,
    scenario,
    tags: List[str],
    map_step: int = 5,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Returns:
      (class_id, stage_name, debug_dict)
    """
    T = len(xs)
    if T < 3:
        return 6, "stage3_short", {"T": T}

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    dist = float(math.hypot(dx, dy))

    h0 = float(headings[0])
    hT = float(headings[-1])
    delta_heading = wrap_to_pi(hT - h0)
    abs_dh = abs(delta_heading)

    dh = np.diff(headings)
    dh = np.vectorize(wrap_to_pi)(dh)
    total_abs = float(np.sum(np.abs(dh)))

    # Stage 2 map hints
    has_intersection_map, has_roundabout_map, lane_turn_class = map_based_scene_flags(scenario, xs, ys, step=map_step)

    # Tag-based intersection hint
    has_intersection = bool(has_intersection_map or tag_intersection_hint(getattr(scenario, "scenario_type", ""), tags))

    debug = {
        "dist": dist,
        "delta_heading_rad": float(delta_heading),
        "delta_heading_deg": float(math.degrees(delta_heading)),
        "abs_delta_heading_deg": float(math.degrees(abs_dh)),
        "total_abs_heading_deg": float(math.degrees(total_abs)),
        "has_intersection_map": bool(has_intersection_map),
        "has_roundabout_map": bool(has_roundabout_map),
        "lane_turn_class": lane_turn_class if lane_turn_class is not None else -1,
        "has_intersection_used": bool(has_intersection),
    }

    # Motion gate (prevents stationary becoming turns)
    if dist < MIN_DIST_ANY:
        return 6, "stage3_motion_gate", debug

    # Roundabout: map wins
    if has_roundabout_map:
        return 4, "stage2_map", debug

    # Explicit lane connector semantics (if available)
    if lane_turn_class in (0, 2, 5):
        return int(lane_turn_class), "stage2_map", debug

    # U-turn: geometry
    if abs(abs_dh - UTURN_CENTER) < UTURN_MARGIN:
        return 5, "stage3_geometry", debug

    # Roundabout fallback: geometry (rare; keep as fallback only)
    if total_abs > ROUNDABOUT_TOTAL_MIN and abs_dh < ROUNDABOUT_NET_MAX:
        return 4, "stage3_geometry", debug

    # Intersection logic: rely primarily on NET heading
    if has_intersection:
        # If ego didn't move enough, do NOT call it a turn (heading jitter otherwise)
        if dist < MIN_DIST_TURN_AT_INTERSECTION:
            if abs_dh <= NET_STRAIGHT_MAX_AT_INTERSECTION:
                return 1, "stage3_intersection_net_heading", debug
            return 6, "stage3_intersection_motion_guard", debug

        # Straight through intersection
        if abs_dh <= NET_STRAIGHT_MAX_AT_INTERSECTION:
            return 1, "stage3_intersection_net_heading", debug

        # Turn at intersection
        if NET_TURN_MIN_AT_INTERSECTION <= abs_dh <= NET_TURN_MAX_AT_INTERSECTION:
            return (0 if delta_heading > 0.0 else 2), "stage3_intersection_net_heading", debug

        return 6, "stage3_intersection_fallback", debug

    # Non-intersection: classify straight vs others (no left/right from curvature)
    straight_net_max = math.radians(15.0)
    straight_total_max = math.radians(25.0)
    if abs_dh <= straight_net_max and total_abs <= straight_total_max:
        return 3, "stage3_nonintersection_straight", debug

    return 6, "stage3_nonintersection_other", debug


# --------------------------------------------------------------------------------------
# Combined classification pipeline
# --------------------------------------------------------------------------------------
def classify_emoe_for_scenario(scenario, map_step: int = 5) -> Tuple[int, str, Dict[str, Any]]:
    """
    Stage 1: tags/strings priority
    Stage 2+3: map + geometry verification
    """
    scenario_type = getattr(scenario, "scenario_type", "")
    tags = get_scenario_tags_if_available(scenario)

    # Stage 1
    emoe_id, stage = stage1_from_tags_and_type(scenario_type, tags)
    if emoe_id is not None:
        return int(emoe_id), str(stage), {"scenario_type": scenario_type, "tags": tags}

    # Stage 2+3
    xs, ys, hs = compute_ego_xyh(scenario)
    emoe_id, stage, debug = classify_emoe_from_map_and_geometry(xs, ys, hs, scenario, tags, map_step=map_step)
    debug = dict(debug)
    debug["scenario_type"] = scenario_type
    debug["tags"] = tags
    return int(emoe_id), stage, debug


# --------------------------------------------------------------------------------------
# nuPlan scenario loading
# --------------------------------------------------------------------------------------
def build_scenarios(split: str, max_scenarios: int, num_workers: int) -> List[Any]:
    """
    Build nuPlan scenarios using ScenarioFilter + SingleMachineParallelExecutor.
    Assumes NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT are set.
    """
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(f"Cannot find DB at {db_root}")

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

    scenarios = builder.get_scenarios(scenario_filter, worker)
    return scenarios


# --------------------------------------------------------------------------------------
# Main: generate scene_labels.jsonl + scene_anchors.npy
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini", help="nuPlan split: mini, trainval, etc.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for labels + anchors")
    parser.add_argument("--Ka", type=int, default=24, help="Anchors per class (KMeans clusters)")
    parser.add_argument("--max_scenarios", type=int, default=-1, help="Limit number of scenarios (-1 = all)")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker threads for scenario loading")
    parser.add_argument("--map_step", type=int, default=5, help="Subsample step for map queries along trajectory")
    parser.add_argument("--min_travel_distance", type=float, default=5.0, help="Min travel dist to include in anchors")
    parser.add_argument("--kmeans_seed", type=int, default=0, help="Random seed for KMeans")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "scene_labels.jsonl"
    anchors_path = out_dir / "scene_anchors.npy"

    print(f"[INFO] split={args.split}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] Ka={args.Ka}")
    print(f"[INFO] max_scenarios={args.max_scenarios}")
    print(f"[INFO] map_step={args.map_step}")
    print(f"[INFO] min_travel_distance(for anchors)={args.min_travel_distance}")

    print(f"[INFO] Loading scenarios...")
    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)
    print(f"[INFO] Loaded {len(scenarios)} scenarios.")

    endpoints_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    class_counts = Counter()
    stage_counts = Counter()

    # Write labels streaming to disk (so you don’t lose everything if interrupted)
    f_labels = labels_path.open("w")

    try:
        for scenario in tqdm(scenarios, desc="Classifying + collecting endpoints"):
            token = scenario.token

            # classify
            emoe_id, stage, debug = classify_emoe_for_scenario(scenario, map_step=args.map_step)
            class_counts[emoe_id] += 1
            stage_counts[stage] += 1

            # compute endpoint for anchors (use GT endpoint only, as in paper)
            xs, ys, hs = compute_ego_xyh(scenario)
            if len(xs) >= 2:
                dist = float(math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])))
            else:
                dist = 0.0

            if dist >= float(args.min_travel_distance):
                endpoint_xy = ego_endpoint_in_ego_frame(xs, ys, hs)
                endpoints_by_class[emoe_id].append(endpoint_xy)

            # write record
            record = {
                "token": token,
                "emoe_class_id": int(emoe_id),
                "emoe_class_name": EMOE_SCENE_TYPES[int(emoe_id)],
                "scenario_type": getattr(scenario, "scenario_type", ""),
                "stage": stage,
                "travel_distance_m": float(dist),
                # keep debug but lightweight
                "debug": {
                    k: debug[k]
                    for k in [
                        "dist",
                        "delta_heading_deg",
                        "abs_delta_heading_deg",
                        "total_abs_heading_deg",
                        "has_intersection_map",
                        "has_roundabout_map",
                        "lane_turn_class",
                        "has_intersection_used",
                        "scenario_type",
                        "tags",
                    ]
                    if k in debug
                },
            }
            f_labels.write(json.dumps(record) + "\n")

    finally:
        f_labels.close()

    print("\n[INFO] Scenario counts per class:")
    for c in range(7):
        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:28s}): {class_counts[c]}")

    print("\n[INFO] Scenario counts per stage:")
    for k, v in stage_counts.most_common():
        print(f"  {k:32s}: {v}")

    print("\n[INFO] Endpoint counts per class (for anchor clustering):")
    for c in range(7):
        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:28s}): {len(endpoints_by_class[c])} endpoints")

    # KMeans per class -> anchors
    Ka = int(args.Ka)
    scene_anchors = np.zeros((7, Ka, 2), dtype=np.float32)

    print("\n[INFO] Running KMeans per class (endpoints only)...")
    for c in range(7):
        pts = np.asarray(endpoints_by_class[c], dtype=np.float32)  # [N,2]
        if pts.shape[0] == 0:
            print(f"[WARN] No endpoints for class {c} ({EMOE_SCENE_TYPES[c]}). Anchors stay zeros.")
            continue

        n_clusters = min(Ka, pts.shape[0])
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=int(args.kmeans_seed),
            n_init="auto" if hasattr(KMeans, "n_init") else 10,
        )
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_.astype(np.float32)

        scene_anchors[c, :n_clusters, :] = centers

        # pad if fewer points than Ka
        if n_clusters < Ka:
            reps = Ka - n_clusters
            scene_anchors[c, n_clusters:, :] = np.repeat(centers[:1, :], reps, axis=0)

        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:28s}): {pts.shape[0]} pts -> {n_clusters} clusters")

    np.save(anchors_path, scene_anchors)
    print(f"\n[INFO] Saved anchors to: {anchors_path}  shape={scene_anchors.shape}")
    print(f"[INFO] Saved labels to:  {labels_path}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
