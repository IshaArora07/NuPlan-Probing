#!/usr/bin/env python3
"""
Precompute EMoE scene labels and scene anchors from nuPlan.

Outputs:
  - scene_labels.jsonl : one line per scenario: token -> emoe_class_id + debug + stage
  - scene_anchors.npy  : shape [7, Ka, 2], KMeans over trajectory endpoints per class

Classification logic (agreed):
  1) Direct tag overrides:
       - right turn tags  -> RIGHT_TURN_AT_INTERSECTION (2)
       - u-turn tags      -> U_TURN (5)
       - roundabout tags  -> ROUNDABOUT (4)
  2) Otherwise:
       - Determine intersection context using LANE_CONNECTOR.intersection_fid (near ego)
       - Determine maneuver using ego geometry (net heading, accumulated turn, etc.)
       - Use connector turn_type_fid as tie-break ONLY when geometry is ambiguous
         (ignore NONE)

Anchors:
  - cluster ONLY trajectory endpoints in the initial ego frame (x,y)

Run (example):
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python emoe_precompute_labels_and_anchors.py \
    --split mini \
    --output_dir /path/to/out \
    --Ka 24 \
    --max_scenarios 20000
"""

import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# clustering
from sklearn.cluster import MiniBatchKMeans


# ------------------------------------------------------------
# EMoE class names (fixed)
# ------------------------------------------------------------
EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
    "others",                         # 6
]


# ------------------------------------------------------------
# Basic utils
# ------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ego rear-axle x, y, heading over the scenario horizon."""
    xs_list: List[float] = []
    ys_list: List[float] = []
    hs_list: List[float] = []
    n = scenario.get_number_of_iterations()
    for i in range(n):
        ego = scenario.get_ego_state_at_iteration(i)
        xs_list.append(ego.rear_axle.x)
        ys_list.append(ego.rear_axle.y)
        hs_list.append(float(ego.rear_axle.heading))
    return (
        np.asarray(xs_list, dtype=np.float64),
        np.asarray(ys_list, dtype=np.float64),
        np.asarray(hs_list, dtype=np.float64),
    )


def endpoint_in_initial_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """
    Endpoint (x,y) in the initial ego frame.
    """
    if len(xs) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    theta0 = float(hs[0])

    c = math.cos(-theta0)
    s = math.sin(-theta0)
    x_rel = c * dx - s * dy
    y_rel = s * dx + c * dy
    return np.array([x_rel, y_rel], dtype=np.float32)


# ------------------------------------------------------------
# Scenario loading (hard cap applied in main)
# ------------------------------------------------------------
def build_scenarios(split: str, max_workers: int = 8) -> List[Any]:
    """
    Load scenarios (may return many; we hard-slice in main with --max_scenarios).
    """
    data_root = os.environ.get("NUPLAN_DATA_ROOT", None)
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", None)
    if data_root is None or map_root is None:
        raise RuntimeError(
            "Please set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT.\n"
            "Example:\n"
            "  export NUPLAN_DATA_ROOT=/path/to/nuplan\n"
            "  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps"
        )

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(f"Cannot find DB at {db_root}")

    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=max_workers)

    # IMPORTANT: ScenarioFilter limit_total_scenarios is not always a strict cap.
    # We'll slice the returned list in main().
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=None,
    )

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=max_workers,
    )

    scenarios = builder.get_scenarios(scenario_filter, worker)
    return scenarios


# ------------------------------------------------------------
# Tag parsing (direct overrides + priors)
# ------------------------------------------------------------
def tag_direct_override(stype: str) -> Optional[Tuple[int, str]]:
    """
    Direct classification from tags.
    Returns (class_id, stage) or None.
    """
    s = stype.upper()

    # Direct roundabout
    if "ROUNDABOUT" in s:
        return 4, "TAG_DIRECT_ROUNDABOUT"

    # Direct U-turn
    if "UTURN" in s or "U_TURN" in s or "U-TURN" in s:
        return 5, "TAG_DIRECT_UTURN"

    # Direct right turn tags (you requested)
    # Keep it strict to avoid accidental matches
    if "STARTING_RIGHT_TURN" in s or "STARTING_RIGHT_TURNS" in s:
        return 2, "TAG_DIRECT_RIGHT_TURN"

    return None


def tag_turn_prior(stype: str) -> Optional[str]:
    """
    Weak priors used only for ambiguity resolution.
    Returns one of: 'LEFT', 'RIGHT', 'STRAIGHT' or None.
    """
    s = stype.upper()
    # cross vs noncross heuristics (weak, dataset dependent)
    if "_CROSS_TURN" in s and "NONCROSS" not in s:
        return "LEFT"
    if "NONCROSS_TURN" in s:
        return "RIGHT"
    if "TRAVERSING_INTERSECTION" in s or "TRAVERSING_TRAFFIC_LIGHT_INTERSECTION" in s:
        return "STRAIGHT"
    return None


# ------------------------------------------------------------
# Map probing helpers (LANE_CONNECTOR layer)
# ------------------------------------------------------------
def get_lane_connector_gdf(map_api):
    """
    Load lane connector vector layer (GeoDataFrame).
    Uses private API like your probing scripts, since it works in your setup.
    """
    try:
        return map_api._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR)  # type: ignore[attr-defined]
    except Exception:
        return None


def nearest_connector_votes_along_trajectory(
    conn_gdf,
    xs: np.ndarray,
    ys: np.ndarray,
    step: int = 5,
    max_dist_m: float = 5.0,
    window_radius_m: float = 80.0,
) -> Tuple[bool, Counter]:
    """
    Along the ego trajectory, sample every `step` frames and query nearest lane connector.
    Returns:
      - has_intersection: True if any matched connector has intersection_fid non-null
      - votes: Counter over connector turn types (strings) INCLUDING 'NONE'
              turn_type_fid mapping: 0 STRAIGHT, 1 LEFT, 2 RIGHT, 3 UTURN, 4 UNKNOWN
    """
    votes = Counter()
    has_intersection = False

    if conn_gdf is None or len(conn_gdf) == 0:
        votes["NONE"] += int(math.ceil(len(xs) / max(1, step)))
        return False, votes

    # Require columns we need
    if "geometry" not in conn_gdf.columns:
        votes["NONE"] += int(math.ceil(len(xs) / max(1, step)))
        return False, votes

    # Determine which columns exist
    has_turn = "turn_type_fid" in conn_gdf.columns
    has_inter = "intersection_fid" in conn_gdf.columns

    # Local window to keep the candidate set smaller
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    minx, maxx = cx - window_radius_m, cx + window_radius_m
    miny, maxy = cy - window_radius_m, cy + window_radius_m
    try:
        local = conn_gdf.cx[minx:maxx, miny:maxy]
    except Exception:
        local = conn_gdf

    if local is None or len(local) == 0:
        votes["NONE"] += int(math.ceil(len(xs) / max(1, step)))
        return False, votes

    # Spatial index nearest query (fast) if available
    try:
        sindex = local.sindex
    except Exception:
        sindex = None

    # Lazy import shapely Point
    from shapely.geometry import Point

    def turn_label_from_val(val) -> str:
        if not has_turn:
            return "UNKNOWN"
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "UNKNOWN"
            iv = int(val)
        except Exception:
            return "UNKNOWN"
        if iv == 0:
            return "STRAIGHT"
        if iv == 1:
            return "LEFT"
        if iv == 2:
            return "RIGHT"
        if iv == 3:
            return "UTURN"
        return "UNKNOWN"

    def has_intersection_from_row(row) -> bool:
        if not has_inter:
            return False
        v = row.get("intersection_fid", None)
        if v is None:
            return False
        try:
            if isinstance(v, float) and math.isnan(v):
                return False
        except Exception:
            pass
        return True

    # Sample trajectory
    for i in range(0, len(xs), step):
        p = Point(float(xs[i]), float(ys[i]))

        best_row = None
        best_dist = max_dist_m

        if sindex is not None:
            # nearest returns candidate indices; we still check exact distance
            try:
                # geopandas sindex.nearest returns generator/array depending on version
                nearest_idx = list(sindex.nearest(p.bounds, return_all=False))[0]
                row = local.iloc[int(nearest_idx)]
                d = float(row.geometry.distance(p))
                if d <= best_dist:
                    best_dist = d
                    best_row = row
            except Exception:
                best_row = None

        # Fallback: brute check only local (still smaller than full)
        if best_row is None:
            try:
                # compute distances vectorized-ish
                dists = local.geometry.distance(p)
                j = int(dists.values.argmin())
                d = float(dists.values[j])
                if d <= best_dist:
                    best_dist = d
                    best_row = local.iloc[j]
            except Exception:
                best_row = None

        if best_row is None:
            votes["NONE"] += 1
            continue

        # Vote
        tl = turn_label_from_val(best_row.get("turn_type_fid", None))
        votes[tl] += 1

        # Intersection context
        if has_intersection_from_row(best_row):
            has_intersection = True

    return has_intersection, votes


def connector_tiebreak_label(votes: Counter) -> Optional[str]:
    """
    Return best connector label among {LEFT, RIGHT, STRAIGHT, UTURN, UNKNOWN},
    IGNORING NONE entirely. If no evidence, return None.
    """
    candidates = ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN"]
    best = None
    best_count = 0
    for k in candidates:
        c = int(votes.get(k, 0))
        if c > best_count:
            best_count = c
            best = k
    return best if best_count > 0 else None


# ------------------------------------------------------------
# Geometry-based maneuver classification
# ------------------------------------------------------------
def geometry_stats(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Dict[str, float]:
    """
    Compute geometry stats used for classification.
    """
    if len(xs) < 2:
        return {"dist": 0.0, "dh_deg": 0.0, "total_abs_deg": 0.0, "pos_deg": 0.0, "neg_deg": 0.0}

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    dist = float(math.hypot(dx, dy))

    dh = wrap_to_pi(float(hs[-1] - hs[0]))
    dh_deg = float(math.degrees(dh))

    dhs = np.diff(hs)
    # wrap each step
    dհ = np.array([wrap_to_pi(float(a)) for a in dhs], dtype=np.float64)
    total_abs = float(np.sum(np.abs(dհ)))
    pos = float(np.sum(np.clip(dհ, 0.0, None)))
    neg = -float(np.sum(np.clip(dհ, None, 0.0)))

    return {
        "dist": dist,
        "dh_deg": dh_deg,
        "total_abs_deg": float(math.degrees(total_abs)),
        "pos_deg": float(math.degrees(pos)),
        "neg_deg": float(math.degrees(neg)),
    }


def geometry_maneuver_label(stats: Dict[str, float]) -> Tuple[str, bool]:
    """
    Decide maneuver from geometry.
    Returns (label, is_ambiguous)

    label in:
      STRAIGHT, LEFT, RIGHT, UTURN, ROUNDABOUT, OTHER
    """
    dist = stats["dist"]
    dh = stats["dh_deg"]
    abs_dh = abs(dh)
    total_abs = stats["total_abs_deg"]
    pos = stats["pos_deg"]
    neg = stats["neg_deg"]

    # distance filter
    if dist < 5.0:
        return "OTHER", True

    # thresholds (tune later if needed)
    straight_net_max = 15.0
    straight_total_max = 25.0

    turn_net_min = 25.0
    turn_net_max = 155.0

    uturn_center = 180.0
    uturn_margin = 35.0

    roundabout_total_min = 270.0
    roundabout_net_max = 90.0

    # U-turn-like
    if abs(abs_dh - uturn_center) < uturn_margin:
        return "UTURN", False

    # Roundabout-like
    if total_abs >= roundabout_total_min and abs_dh <= roundabout_net_max:
        return "ROUNDABOUT", False

    # Straight-like
    if abs_dh <= straight_net_max and total_abs <= straight_total_max:
        return "STRAIGHT", False

    # Left/Right-like
    if turn_net_min <= abs_dh <= turn_net_max:
        if pos >= neg:
            return "LEFT", False
        else:
            return "RIGHT", False

    # Ambiguous region (gentle curves / weird cases)
    # If it has substantial turning but not cleanly a turn class:
    if total_abs > 40.0:
        # pick direction by dominance but mark ambiguous
        if pos > neg:
            return "LEFT", True
        elif neg > pos:
            return "RIGHT", True
        else:
            return "OTHER", True

    return "OTHER", True


# ------------------------------------------------------------
# Final classification (stages)
# ------------------------------------------------------------
def classify_emoe_scenario(
    scenario,
    conn_gdf,
    connector_step: int,
    connector_max_dist_m: float,
    connector_window_radius_m: float,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Returns:
      (emoe_class_id, stage, debug_dict)
    """
    stype = scenario.scenario_type

    # Stage 1: direct tag overrides
    direct = tag_direct_override(stype)
    if direct is not None:
        cid, stage = direct
        return cid, stage, {"scenario_type": stype}

    # Extract geometry
    xs, ys, hs = compute_ego_xyh(scenario)
    stats = geometry_stats(xs, ys, hs)
    geom_label, geom_amb = geometry_maneuver_label(stats)

    # Connector votes + intersection context via intersection_fid
    has_intersection, votes = nearest_connector_votes_along_trajectory(
        conn_gdf,
        xs,
        ys,
        step=connector_step,
        max_dist_m=connector_max_dist_m,
        window_radius_m=connector_window_radius_m,
    )
    conn_hint = connector_tiebreak_label(votes)  # ignores NONE

    # Special cases from geometry if tags did not catch
    if geom_label == "ROUNDABOUT":
        return 4, "GEOMETRY_SPECIAL_ROUNDABOUT", {
            "scenario_type": stype, **stats,
            "has_intersection": bool(has_intersection),
            "connector_hint": conn_hint, "connector_votes": dict(votes),
        }
    if geom_label == "UTURN":
        return 5, "GEOMETRY_SPECIAL_UTURN", {
            "scenario_type": stype, **stats,
            "has_intersection": bool(has_intersection),
            "connector_hint": conn_hint, "connector_votes": dict(votes),
        }

    # Intersection context used for EMoE semantics
    context = "INTERSECTION" if has_intersection else "NON_INTERSECTION"

    # If geometry confident, decide immediately
    if not geom_amb:
        if geom_label == "STRAIGHT":
            cid = 1 if context == "INTERSECTION" else 3
            return cid, "GEOMETRY_MAP", {
                "scenario_type": stype, **stats,
                "has_intersection": bool(has_intersection),
                "connector_hint": conn_hint, "connector_votes": dict(votes),
            }
        if geom_label == "LEFT":
            cid = 0 if context == "INTERSECTION" else 6
            return cid, "GEOMETRY_MAP", {
                "scenario_type": stype, **stats,
                "has_intersection": bool(has_intersection),
                "connector_hint": conn_hint, "connector_votes": dict(votes),
            }
        if geom_label == "RIGHT":
            cid = 2 if context == "INTERSECTION" else 6
            return cid, "GEOMETRY_MAP", {
                "scenario_type": stype, **stats,
                "has_intersection": bool(has_intersection),
                "connector_hint": conn_hint, "connector_votes": dict(votes),
            }
        # OTHER
        return 6, "GEOMETRY_MAP", {
            "scenario_type": stype, **stats,
            "has_intersection": bool(has_intersection),
            "connector_hint": conn_hint, "connector_votes": dict(votes),
        }

    # Stage 3: ambiguity resolution (use priors + connector hint)
    prior = tag_turn_prior(stype)

    # decide direction among LEFT/RIGHT/STRAIGHT from best evidence
    decision = None

    # 1) if connector hint provides direction
    if conn_hint in ("LEFT", "RIGHT", "STRAIGHT"):
        decision = conn_hint
        stage = "AMBIG_TIEBREAK_CONNECTOR"
    # 2) else use tag prior
    elif prior in ("LEFT", "RIGHT", "STRAIGHT"):
        decision = prior
        stage = "AMBIG_TIEBREAK_TAG"
    # 3) else fallback to geometry label if it at least picked a direction
    elif geom_label in ("LEFT", "RIGHT", "STRAIGHT"):
        decision = geom_label
        stage = "AMBIG_FALLBACK_GEOMETRY"
    else:
        decision = "OTHER"
        stage = "FALLBACK_OTHERS"

    if decision == "STRAIGHT":
        cid = 1 if context == "INTERSECTION" else 3
    elif decision == "LEFT":
        cid = 0 if context == "INTERSECTION" else 6
    elif decision == "RIGHT":
        cid = 2 if context == "INTERSECTION" else 6
    else:
        cid = 6

    return cid, stage, {
        "scenario_type": stype, **stats,
        "has_intersection": bool(has_intersection),
        "geom_label": geom_label,
        "connector_hint": conn_hint,
        "tag_prior": prior,
        "connector_votes": dict(votes),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Ka", type=int, default=24)
    parser.add_argument("--max_scenarios", type=int, default=20000,
                        help="Hard cap on scenarios processed. Set -1 for all (not recommended).")

    # connector sampling / matching
    parser.add_argument("--connector_step", type=int, default=5,
                        help="Sample every N frames for connector queries.")
    parser.add_argument("--connector_max_dist_m", type=float, default=5.0,
                        help="Max distance to accept nearest connector as 'hit'.")
    parser.add_argument("--connector_window_radius_m", type=float, default=80.0,
                        help="Local window radius around ego mean for connector layer cropping.")

    # include endpoints only if traveled enough
    parser.add_argument("--min_travel_distance_m", type=float, default=5.0)

    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "scene_labels.jsonl"
    anchors_path = out_dir / "scene_anchors.npy"

    print(f"[INFO] split={args.split}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] Ka={args.Ka}")
    print(f"[INFO] max_scenarios={args.max_scenarios}")

    print("[INFO] Loading scenarios ...")
    scenarios = build_scenarios(args.split, max_workers=8)
    print(f"[INFO] Loaded {len(scenarios)} scenarios total (before hard cap).")

    if args.max_scenarios is not None and args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]
        print(f"[INFO] Restricting processing to first {len(scenarios)} scenarios (hard cap).")
    elif args.max_scenarios == -1:
        print("[WARN] max_scenarios=-1, processing ALL loaded scenarios (may take a long time).")

    # We will lazily load connector layer from first scenario's map_api.
    # (All scenarios in a map should share the same schema.)
    conn_gdf = None
    try:
        conn_gdf = get_lane_connector_gdf(scenarios[0].map_api)
    except Exception:
        conn_gdf = None
    print(f"[INFO] Lane connector layer loaded: {conn_gdf is not None} (rows={0 if conn_gdf is None else len(conn_gdf)})")
    if conn_gdf is not None:
        cols = list(conn_gdf.columns)
        print(f"[INFO] LANE_CONNECTOR columns include: intersection_fid={'intersection_fid' in cols}, turn_type_fid={'turn_type_fid' in cols}")

    # Stats containers
    class_counts = Counter()
    stage_counts = Counter()
    endpoints_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)

    # Write labels
    n_written = 0
    with labels_path.open("w") as f:
        for scenario in tqdm(scenarios, desc="Classifying scenarios"):
            try:
                cid, stage, debug = classify_emoe_scenario(
                    scenario,
                    conn_gdf,
                    connector_step=args.connector_step,
                    connector_max_dist_m=args.connector_max_dist_m,
                    connector_window_radius_m=args.connector_window_radius_m,
                )
            except Exception as e:
                # If something goes wrong, dump to others
                cid, stage, debug = 6, "EXCEPTION_FALLBACK", {"error": str(e), "scenario_type": getattr(scenario, "scenario_type", "UNKNOWN")}

            class_counts[cid] += 1
            stage_counts[stage] += 1

            # endpoint collection for anchors (only if traveled enough)
            try:
                xs, ys, hs = compute_ego_xyh(scenario)
                # travel distance check
                dist = float(math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))) if len(xs) > 1 else 0.0
                if dist >= args.min_travel_distance_m:
                    ep = endpoint_in_initial_ego_frame(xs, ys, hs)
                    endpoints_by_class[cid].append(ep)
            except Exception:
                pass

            record = {
                "token": scenario.token,
                "emoe_class_id": int(cid),
                "emoe_class_name": EMOE_SCENE_TYPES[int(cid)],
                "stage": stage,
                **debug,
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"\n[INFO] Wrote labels for {n_written} scenarios -> {labels_path}")

    # Print stats: per class
    print("\n[STATS] Scenarios per EMoE class:")
    total = sum(class_counts.values())
    for c in range(len(EMOE_SCENE_TYPES)):
        n = class_counts.get(c, 0)
        pct = 100.0 * n / max(1, total)
        print(f"  {c} {EMOE_SCENE_TYPES[c]:28s}: {n:8d} ({pct:5.2f}%)")

    # Print stats: per stage
    print("\n[STATS] Scenarios per decision stage:")
    for k, v in stage_counts.most_common():
        pct = 100.0 * v / max(1, total)
        print(f"  {k:28s}: {v:8d} ({pct:5.2f}%)")

    # Endpoint stats (for anchors)
    print("\n[STATS] Endpoints collected per class (for anchor clustering):")
    for c in range(len(EMOE_SCENE_TYPES)):
        n = len(endpoints_by_class.get(c, []))
        print(f"  {c} {EMOE_SCENE_TYPES[c]:28s}: {n:8d} endpoints")

    # ------------------------------------------------------------
    # KMeans anchors per class (endpoints only)
    # ------------------------------------------------------------
    num_classes = len(EMOE_SCENE_TYPES)
    Ka = int(args.Ka)
    scene_anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)

    print("\n[INFO] Running MiniBatchKMeans per class to compute anchors ...")
    for c in range(num_classes):
        pts = np.asarray(endpoints_by_class.get(c, []), dtype=np.float32)  # [N,2]
        npts = int(pts.shape[0])

        if npts == 0:
            print(f"  [WARN] Class {c} has 0 endpoints. Anchors remain zeros.")
            continue

        k = min(Ka, npts)
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=0,
            batch_size=4096,
            n_init="auto" if hasattr(MiniBatchKMeans, "n_init") else 10,
        )
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_.astype(np.float32)  # [k,2]

        scene_anchors[c, :k, :] = centers
        if k < Ka:
            # fill remaining by repeating first center (deterministic)
            scene_anchors[c, k:, :] = centers[:1, :]

        print(f"  [INFO] Class {c} ({EMOE_SCENE_TYPES[c]}): {npts} pts -> {k} clusters")

    np.save(anchors_path, scene_anchors)
    print(f"\n[INFO] Saved anchors -> {anchors_path}")
    print(f"[INFO] scene_anchors shape: {scene_anchors.shape}  (7, Ka, 2)")

    print("\n[DONE] Outputs:")
    print(f"  - {labels_path}")
    print(f"  - {anchors_path}")


if __name__ == "__main__":
    main()
