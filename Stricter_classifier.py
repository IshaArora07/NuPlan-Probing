#!/usr/bin/env python3
"""
Precompute EMoE scene labels + scene anchors from nuPlan.

Outputs (in --output_dir):
  - scene_labels.jsonl  : one line per scenario (token -> emoe_class_id + debug + stage_used)
  - scene_anchors.npy   : shape [7, Ka, 2] (KMeans on GT trajectory endpoints per class)

Pipeline (priority stages):
  Stage 1 (tags/strings priority):
    - roundabout -> class 4
    - u-turn     -> class 5
    - starting_right_turn / right_turn -> class 2   (DIRECT)
    - intersection tags -> do NOT directly left/right here; geometry decides later

  Stage 2 (SemanticMapLayer-based map semantics):
    - Intersection traversal evidence using INTERSECTION polygons + buffering/tolerance
    - Connector evidence using LANE_CONNECTOR lines + turn_type_fid (votes near ego points)

  Stage 3 (geometry decision; STRICT but avoids over-"others"):
    - HARD motion gate: stationary/slow-jitter can’t become LEFT/RIGHT
    - Compute INTERSECTION CONTEXT with 3 tiers:
        Tier-1: ego actually traverses intersection (inside/near hits + distance through)
        Tier-2: intersection likely even if polygon misses (2-of-3 evidence: tag, connector, near)
        Tier-3: no intersection context
    - If intersection context:
        net heading decides LEFT/RIGHT/STRAIGHT
        + optional "window-clipped turn rescue" using total_abs_heading (only with strong intersection evidence)
    - Else:
        STRAIGHT_NON_INTERSECTION vs OTHERS only

Notes on GPU:
  - This script is I/O + GeoPandas/Shapely bound. GPU won’t meaningfully speed it up.

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python precompute_emoe_labels_anchors.py \
    --split mini \
    --output_dir ./emoe_precomputed_mini \
    --Ka 24 \
    --max_scenarios 20000
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
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# shapely
from shapely.geometry import Point


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
# Geometry thresholds / gates
# --------------------------------------------------------------------------------------
MIN_DIST_ANY = 5.0
MIN_DIST_TURN_AT_INTERSECTION = 12.0

# Intersection net-heading thresholds (YOU asked: require > 35° for turn)
NET_STRAIGHT_MAX_AT_INTERSECTION = math.radians(12.0)
NET_TURN_MIN_AT_INTERSECTION = math.radians(35.0)
NET_TURN_MAX_AT_INTERSECTION = math.radians(165.0)

UTURN_CENTER = math.pi
UTURN_MARGIN = math.radians(35.0)

# Non-intersection straightness gate
NONINT_STRAIGHT_NET_MAX = math.radians(15.0)
NONINT_STRAIGHT_TOTAL_MAX = math.radians(25.0)

# --- NEW: intersection traversal evidence + strictness knobs ---
INTERSECTION_TIER1_MIN_HITS = 2              # at least these many "inside/near" hits
INTERSECTION_TIER1_MIN_DIST_THROUGH = 8.0    # must travel this much while in/near intersection
INTERSECTION_TIER2_NEAR_TOL_M = 18.0         # looser tol for Tier-2 "near" evidence
CONNECTOR_STRONG_RATIO = 0.55                # best_ratio threshold for "strong" connector evidence
CONNECTOR_STRONG_MIN_COUNT = 3               # best_count threshold (ignore NONE)
TURN_RESCUE_TOTAL_ABS_MIN = math.radians(60.0)   # rescue if total_abs heading big enough
TURN_RESCUE_DOMINANCE = 0.65                     # consistent turning sign ratio


# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, hs = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(ego.rear_axle.x)
        ys.append(ego.rear_axle.y)
        hs.append(float(ego.rear_axle.heading))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(hs, dtype=np.float64)


def ego_endpoint_in_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> np.ndarray:
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
    tags: List[str] = []
    for attr in ["tags", "scenario_tags", "log_tags"]:
        if hasattr(scenario, attr):
            try:
                val = getattr(scenario, attr)
                if isinstance(val, (list, tuple)):
                    tags.extend([_upper(t) for t in val])
            except Exception:
                pass

    for method in ["get_tags", "get_scenario_tags"]:
        if hasattr(scenario, method):
            try:
                val = getattr(scenario, method)()
                if isinstance(val, (list, tuple)):
                    tags.extend([_upper(t) for t in val])
            except Exception:
                pass

    seen = set()
    out = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def stage1_from_tags_and_type(scenario_type: str, tags: List[str]) -> Tuple[Optional[int], Optional[str]]:
    st = _upper(scenario_type)
    tset = set(tags)

    if "ROUNDABOUT" in st or any("ROUNDABOUT" in t for t in tset):
        return 4, "stage1_tags"

    if "UTURN" in st or "U_TURN" in st or any(("UTURN" in t or "U_TURN" in t) for t in tset):
        return 5, "stage1_tags"

    # DIRECT right-turn (your requirement)
    if (
        ("STARTING_RIGHT" in st and "TURN" in st)
        or ("RIGHT_TURN" in st)
        or any(("STARTING_RIGHT" in t and "TURN" in t) for t in tset)
        or any(("RIGHT" in t and "TURN" in t and "STARTING" in t) for t in tset)
    ):
        return 2, "stage1_tags"

    return None, None


def tag_intersection_hint(scenario_type: str, tags: List[str]) -> bool:
    st = _upper(scenario_type)
    tset = set(tags)
    keys = [
        "INTERSECTION",
        "TRAFFIC_LIGHT",
        "STOP_SIGN",
        "TRAVERSING_INTERSECTION",
        "TRAVERSING_TRAFFIC_LIGHT_INTERSECTION",
        "ON_INTERSECTION",
        "ON_TRAFFIC_LIGHT_INTERSECTION",
    ]
    if any(k in st for k in keys):
        return True
    if any(any(k in t for k in keys) for t in tset):
        return True
    return False


# --------------------------------------------------------------------------------------
# Stage 2: SemanticMapLayer-based map semantics
# --------------------------------------------------------------------------------------
def _get_vector_map_layer(map_api, layer: SemanticMapLayer):
    try:
        return map_api._get_vector_map_layer(layer)  # type: ignore[attr-defined]
    except Exception:
        return None


def _safe_sindex(gdf):
    try:
        return gdf.sindex
    except Exception:
        return None


# ------------------------- NEW: intersection traversal evidence -------------------------
def intersection_traversal_evidence(
    map_api,
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    tol_m: float,
) -> Dict[str, Any]:
    """
    Compute robust intersection evidence using INTERSECTION polygons.

    Returns:
      {
        "has_any": bool,
        "min_dist_m": float,
        "hits": int,
        "hit_indices": List[int],
        "dist_through_m": float,      # distance traveled while points are in/near intersection
      }

    Implementation details:
      - Uses spatial index if available
      - Uses bbox query and then exact distance/contains
    """
    inter_gdf = _get_vector_map_layer(map_api, SemanticMapLayer.INTERSECTION)
    if inter_gdf is None or len(inter_gdf) == 0:
        return {
            "has_any": False,
            "min_dist_m": float("inf"),
            "hits": 0,
            "hit_indices": [],
            "dist_through_m": 0.0,
        }

    sindex = _safe_sindex(inter_gdf)
    r = float(tol_m)

    min_dist = float("inf")
    hit_indices: List[int] = []

    # precompute per-step points (avoid repeated float conversions)
    sample_indices = list(range(0, len(xs), max(1, int(sample_step))))
    points = [Point(float(xs[i]), float(ys[i])) for i in sample_indices]

    for si, p in zip(sample_indices, points):
        # candidate polygons
        if sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(sindex.intersection(bbox))
            cand = inter_gdf.iloc[cand_idx] if cand_idx else None
        else:
            cand = inter_gdf

        if cand is None or len(cand) == 0:
            continue

        hit_here = False
        for geom in cand.geometry:
            if geom is None:
                continue
            try:
                if geom.contains(p):
                    min_dist = 0.0
                    hit_here = True
                    break
                d = float(geom.distance(p))
                if d < min_dist:
                    min_dist = d
                if d <= r:
                    hit_here = True
                    break
            except Exception:
                continue

        if hit_here:
            hit_indices.append(int(si))

    # distance traveled while in/near intersection (use consecutive hits along original time axis)
    dist_through = 0.0
    if len(hit_indices) >= 2:
        hit_set = set(hit_indices)
        prev = None
        for i in range(1, len(xs)):
            if i in hit_set:
                if prev is None:
                    prev = i
                else:
                    # accumulate segment from prev->i (approx; good enough)
                    dx = float(xs[i] - xs[prev])
                    dy = float(ys[i] - ys[prev])
                    dist_through += float(math.hypot(dx, dy))
                    prev = i

    has_any = (len(hit_indices) > 0) or (min_dist <= r)

    return {
        "has_any": bool(has_any),
        "min_dist_m": float(min_dist),
        "hits": int(len(hit_indices)),
        "hit_indices": hit_indices,
        "dist_through_m": float(dist_through),
    }


# ------------------------- connector voting (your existing idea, plus best_count) -------------------------
def connector_vote_from_layer(
    map_api,
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    connector_radius_m: float,
) -> Dict[str, Any]:
    conn_gdf = _get_vector_map_layer(map_api, SemanticMapLayer.LANE_CONNECTOR)
    if conn_gdf is None or len(conn_gdf) == 0:
        return {
            "counts": {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0, "UTURN": 0, "UNKNOWN": 0, "NONE": 0},
            "total_samples": 0,
            "best_type": "NONE",
            "best_ratio": 0.0,
            "best_count": 0,
            "has_connector_evidence": False,
            "turn_type_col": "",
        }

    col = None
    for c in ["turn_type_fid", "lane_connector_type_fid", "turn_type", "lane_connector_type"]:
        if c in conn_gdf.columns:
            col = c
            break

    sindex = _safe_sindex(conn_gdf)
    counts = Counter()
    total = 0

    def to_label(v) -> str:
        # LaneConnectorType: STRAIGHT=0, LEFT=1, RIGHT=2, UTURN=3, UNKNOWN=4
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "UNKNOWN"
            iv = int(v)
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

    r = float(connector_radius_m)

    for i in range(0, len(xs), max(1, int(sample_step))):
        p = Point(float(xs[i]), float(ys[i]))
        total += 1

        if sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(sindex.intersection(bbox))
            cand = conn_gdf.iloc[cand_idx] if cand_idx else None
        else:
            cand = conn_gdf

        if cand is None or len(cand) == 0:
            counts["NONE"] += 1
            continue

        any_hit = False
        for _, row in cand.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            try:
                d = float(geom.distance(p))
            except Exception:
                continue
            if d <= r:
                any_hit = True
                v = row[col] if (col is not None and col in row) else None
                counts[to_label(v)] += 1

        if not any_hit:
            counts["NONE"] += 1

    # choose best ignoring NONE (your preference)
    best_type = "NONE"
    best_count = 0
    for k in ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN"]:
        if counts[k] > best_count:
            best_count = counts[k]
            best_type = k

    denom = sum(counts[k] for k in ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN"])
    best_ratio = (best_count / max(1, int(denom)))
    has_connector_evidence = (best_count > 0)

    return {
        "counts": {
            "LEFT": int(counts["LEFT"]),
            "RIGHT": int(counts["RIGHT"]),
            "STRAIGHT": int(counts["STRAIGHT"]),
            "UTURN": int(counts["UTURN"]),
            "UNKNOWN": int(counts["UNKNOWN"]),
            "NONE": int(counts["NONE"]),
        },
        "total_samples": int(total),
        "best_type": str(best_type),
        "best_ratio": float(best_ratio),
        "best_count": int(best_count),
        "has_connector_evidence": bool(has_connector_evidence),
        "turn_type_col": col if col is not None else "",
    }


# --------------------------------------------------------------------------------------
# Stage 3: STRICT but avoids too-many-others (3-tier intersection context + rescue)
# --------------------------------------------------------------------------------------
def classify_with_priority_strict(
    scenario,
    xs: np.ndarray,
    ys: np.ndarray,
    headings: np.ndarray,
    tags: List[str],
    *,
    map_sample_step: int,
    intersection_tol_m: float,
    connector_sample_step: int,
    connector_radius_m: float,
) -> Tuple[int, str, Dict[str, Any]]:
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

    # sign consistency for "window clipped" rescue
    # (ignore near-zero jitters)
    dh_small = dh[np.abs(dh) > math.radians(1.0)]
    if dh_small.size > 0:
        pos_ratio = float(np.mean(dh_small > 0.0))
        neg_ratio = float(np.mean(dh_small < 0.0))
    else:
        pos_ratio, neg_ratio = 0.5, 0.5

    # motion gate
    if dist < MIN_DIST_ANY:
        return 6, "stage3_motion_gate", {
            "dist": dist,
            "delta_heading_deg": float(math.degrees(delta_heading)),
            "abs_delta_heading_deg": float(math.degrees(abs_dh)),
            "total_abs_heading_deg": float(math.degrees(total_abs)),
        }

    map_api = getattr(scenario, "map_api", None)

    # tag hint
    has_intersection_tag = tag_intersection_hint(getattr(scenario, "scenario_type", ""), tags)

    # --- intersection evidence (Tier-1 uses strict tol; Tier-2 can use looser tol) ---
    inter_ev = {
        "has_any": False,
        "min_dist_m": float("inf"),
        "hits": 0,
        "hit_indices": [],
        "dist_through_m": 0.0,
    }
    inter_ev_loose = {
        "has_any": False,
        "min_dist_m": float("inf"),
        "hits": 0,
        "hit_indices": [],
        "dist_through_m": 0.0,
    }
    if map_api is not None:
        inter_ev = intersection_traversal_evidence(
            map_api, xs, ys, sample_step=map_sample_step, tol_m=float(intersection_tol_m)
        )
        # looser evidence only for Tier-2
        inter_ev_loose = intersection_traversal_evidence(
            map_api, xs, ys, sample_step=map_sample_step, tol_m=float(INTERSECTION_TIER2_NEAR_TOL_M)
        )

    # connector vote (verification + Tier-2 evidence)
    conn_info = {
        "counts": {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0, "UTURN": 0, "UNKNOWN": 0, "NONE": 0},
        "total_samples": 0,
        "best_type": "NONE",
        "best_ratio": 0.0,
        "best_count": 0,
        "has_connector_evidence": False,
        "turn_type_col": "",
    }
    if map_api is not None:
        conn_info = connector_vote_from_layer(
            map_api, xs, ys, sample_step=connector_sample_step, connector_radius_m=connector_radius_m
        )

    # --- u-turn (geometry) ---
    if abs(abs_dh - UTURN_CENTER) < UTURN_MARGIN:
        debug = {
            "dist": dist,
            "delta_heading_deg": float(math.degrees(delta_heading)),
            "abs_delta_heading_deg": float(math.degrees(abs_dh)),
            "total_abs_heading_deg": float(math.degrees(total_abs)),
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "has_intersection_tag": bool(has_intersection_tag),
            "intersection": {"strict": inter_ev, "loose": inter_ev_loose},
            "connector": conn_info,
        }
        return 5, "stage3_geometry_uturn", debug

    # --- NEW: compute tiered intersection context ---
    # Tier-1: traverses intersection (strict)
    tier1 = (
        bool(inter_ev["hits"] >= INTERSECTION_TIER1_MIN_HITS)
        and float(inter_ev["dist_through_m"]) >= float(INTERSECTION_TIER1_MIN_DIST_THROUGH)
    )

    # Tier-2 evidence items:
    e_tag = bool(has_intersection_tag)
    e_near = bool(inter_ev_loose["has_any"])  # loose near/inside evidence
    e_conn_strong = bool(
        conn_info.get("has_connector_evidence", False)
        and conn_info.get("best_type", "NONE") in ("LEFT", "RIGHT", "STRAIGHT", "UTURN")
        and float(conn_info.get("best_ratio", 0.0)) >= float(CONNECTOR_STRONG_RATIO)
        and int(conn_info.get("best_count", 0)) >= int(CONNECTOR_STRONG_MIN_COUNT)
    )
    tier2 = (sum([e_tag, e_near, e_conn_strong]) >= 2)

    # Tier-3 otherwise
    has_intersection_context = bool(tier1 or tier2)

    context_source = (
        "tier1_traversal"
        if tier1
        else ("tier2_2of3" if tier2 else "tier3_none")
    )

    debug = {
        "dist": dist,
        "delta_heading_deg": float(math.degrees(delta_heading)),
        "abs_delta_heading_deg": float(math.degrees(abs_dh)),
        "total_abs_heading_deg": float(math.degrees(total_abs)),
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "has_intersection_tag": bool(has_intersection_tag),
        "tier1_traversal": bool(tier1),
        "tier2_evidence": {"tag": e_tag, "near": e_near, "connector_strong": e_conn_strong},
        "tier2_2of3": bool(tier2),
        "context_source": context_source,
        "intersection": {"strict": inter_ev, "loose": inter_ev_loose},
        "connector": conn_info,
    }

    # --- If intersection context: geometry decides direction ---
    if has_intersection_context:
        # strict: only call straight_at_intersection if ego actually "moves through" intersection.
        # Tier-1 always qualifies. Tier-2 qualifies only if some traversal-ish distance exists.
        traversalish = bool(
            tier1 or (float(inter_ev_loose["dist_through_m"]) >= 5.0) or (float(dist) >= 20.0)
        )

        # guard: avoid stationary false turns inside junction
        if dist < MIN_DIST_TURN_AT_INTERSECTION:
            if abs_dh <= NET_STRAIGHT_MAX_AT_INTERSECTION and traversalish:
                return 1, "stage3_intersection_straight_motion_guard", debug
            return 6, "stage3_intersection_motion_guard", debug

        # Straight through intersection (ONLY if traversalish)
        if abs_dh <= NET_STRAIGHT_MAX_AT_INTERSECTION and traversalish:
            return 1, "stage3_intersection_net_heading_straight", debug

        # Turn at intersection by net heading
        if NET_TURN_MIN_AT_INTERSECTION <= abs_dh <= NET_TURN_MAX_AT_INTERSECTION:
            return (0 if delta_heading > 0.0 else 2), "stage3_intersection_net_heading_turn", debug

        # --- NEW: window-clipped turn rescue (ONLY with strong intersection evidence) ---
        # If net heading < 35° but total_abs is high and heading sign is consistent,
        # we allow it as turn AT INTERSECTION (still not in non-intersection).
        strong_for_rescue = bool(tier1 or e_conn_strong or e_tag)
        consistent_left = (pos_ratio >= TURN_RESCUE_DOMINANCE)
        consistent_right = (neg_ratio >= TURN_RESCUE_DOMINANCE)

        if strong_for_rescue and (total_abs >= TURN_RESCUE_TOTAL_ABS_MIN) and (consistent_left or consistent_right):
            return (0 if consistent_left else 2), "stage3_intersection_turn_rescue_total_abs", debug

        # otherwise fallback
        return 6, "stage3_intersection_fallback", debug

    # --- No intersection context: only straight_non_intersection vs others ---
    if abs_dh <= NONINT_STRAIGHT_NET_MAX and total_abs <= NONINT_STRAIGHT_TOTAL_MAX:
        return 3, "stage3_nonintersection_straight", debug

    return 6, "stage3_nonintersection_other", debug


# --------------------------------------------------------------------------------------
# Combined classification pipeline
# --------------------------------------------------------------------------------------
def classify_emoe_for_scenario(
    scenario,
    *,
    map_sample_step: int,
    intersection_tol_m: float,
    connector_sample_step: int,
    connector_radius_m: float,
) -> Tuple[int, str, Dict[str, Any]]:
    scenario_type = getattr(scenario, "scenario_type", "")
    tags = get_scenario_tags_if_available(scenario)

    # Stage 1 direct
    emoe_id, stage = stage1_from_tags_and_type(scenario_type, tags)
    if emoe_id is not None:
        return int(emoe_id), str(stage), {"scenario_type": scenario_type, "tags": tags}

    # Stage 2+3 strict priority
    xs, ys, hs = compute_ego_xyh(scenario)
    emoe_id, stage, debug = classify_with_priority_strict(
        scenario, xs, ys, hs, tags,
        map_sample_step=map_sample_step,
        intersection_tol_m=intersection_tol_m,
        connector_sample_step=connector_sample_step,
        connector_radius_m=connector_radius_m,
    )
    debug = dict(debug)
    debug["scenario_type"] = scenario_type
    debug["tags"] = tags
    return int(emoe_id), stage, debug


# --------------------------------------------------------------------------------------
# nuPlan scenario loading
# --------------------------------------------------------------------------------------
def build_scenarios(split: str, max_scenarios: int, num_workers: int) -> List[Any]:
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

    # knobs for SemanticMapLayer intersection/connector logic
    parser.add_argument("--map_sample_step", type=int, default=2, help="Subsample step for INTERSECTION checks (2 recommended)")
    parser.add_argument("--intersection_tol_m", type=float, default=12.0, help="STRICT distance-to-intersection tol [m] (Tier-1)")
    parser.add_argument("--connector_sample_step", type=int, default=2, help="Subsample step for connector voting")
    parser.add_argument("--connector_radius_m", type=float, default=5.0, help="Radius to count connectors near ego [m]")

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
    print(f"[INFO] map_sample_step={args.map_sample_step}")
    print(f"[INFO] intersection_tol_m={args.intersection_tol_m} (Tier-1 strict)")
    print(f"[INFO] connector_sample_step={args.connector_sample_step}")
    print(f"[INFO] connector_radius_m={args.connector_radius_m}")
    print(f"[INFO] min_travel_distance(for anchors)={args.min_travel_distance}")

    print(f"[INFO] Loading scenarios...")
    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)

    # hard cap so tqdm denominator matches
    if args.max_scenarios is not None and args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    print(f"[INFO] Loaded {len(scenarios)} scenarios (after hard cap).")

    endpoints_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    class_counts = Counter()
    stage_counts = Counter()

    with labels_path.open("w") as f_labels:
        for scenario in tqdm(scenarios, total=len(scenarios), desc="Classifying + collecting endpoints"):
            token = scenario.token

            emoe_id, stage, debug = classify_emoe_for_scenario(
                scenario,
                map_sample_step=args.map_sample_step,
                intersection_tol_m=args.intersection_tol_m,
                connector_sample_step=args.connector_sample_step,
                connector_radius_m=args.connector_radius_m,
            )
            class_counts[emoe_id] += 1
            stage_counts[stage] += 1

            xs, ys, hs = compute_ego_xyh(scenario)
            dist = float(math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))) if len(xs) >= 2 else 0.0

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
                "debug": {
                    "dist": debug.get("dist", None),
                    "delta_heading_deg": debug.get("delta_heading_deg", None),
                    "abs_delta_heading_deg": debug.get("abs_delta_heading_deg", None),
                    "total_abs_heading_deg": debug.get("total_abs_heading_deg", None),
                    "pos_ratio": debug.get("pos_ratio", None),
                    "neg_ratio": debug.get("neg_ratio", None),
                    "has_intersection_tag": debug.get("has_intersection_tag", None),
                    "tier1_traversal": debug.get("tier1_traversal", None),
                    "tier2_evidence": debug.get("tier2_evidence", None),
                    "tier2_2of3": debug.get("tier2_2of3", None),
                    "context_source": debug.get("context_source", None),
                    "intersection_strict": (debug.get("intersection", {}) or {}).get("strict", None),
                    "intersection_loose": (debug.get("intersection", {}) or {}).get("loose", None),
                    "connector_best_type": (debug.get("connector", {}) or {}).get("best_type", None),
                    "connector_best_ratio": (debug.get("connector", {}) or {}).get("best_ratio", None),
                    "connector_best_count": (debug.get("connector", {}) or {}).get("best_count", None),
                    "connector_counts": (debug.get("connector", {}) or {}).get("counts", None),
                    "scenario_type": debug.get("scenario_type", None),
                    "tags": debug.get("tags", None),
                },
            }
            f_labels.write(json.dumps(record) + "\n")

    print("\n[INFO] Scenario counts per class:")
    for c in range(7):
        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:28s}): {class_counts[c]}")

    print("\n[INFO] Scenario counts per stage:")
    for k, v in stage_counts.most_common():
        print(f"  {k:44s}: {v}")

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
