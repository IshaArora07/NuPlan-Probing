#!/usr/bin/env python3
"""
Precompute EMoE scene labels + scene anchors from nuPlan (STRICT traversal)
with lane-following alignment + improved U-turn/roundabout intent logic
(to reduce "others" and keep ego-lane-aware rule-based behavior).

Outputs (in --output_dir):
  - scene_labels.jsonl  : one line per scenario (token -> emoe_class_id + debug + stage_used)
  - scene_anchors.npy   : shape [7, Ka, 2] (KMeans on GT trajectory endpoints per class)

Key behavior:
  ✅ "at_intersection" classes ONLY if ego actually TRAVERSES the intersection:
        - intersects INTERSECTION polygons along trajectory, OR
        - traverses a LANE_CONNECTOR (map-matched along ego path)
  ✅ LEFT/RIGHT/STRAIGHT at intersection decided primarily by NET heading (geometry),
     connector turn_type used only as verification.
  ✅ Lane-following alignment (ego heading vs lane/connector tangent)
        - fixes intersection gap (12°–35°): can become straight_at_intersection
        - fixes curved lane-following outside intersections: can become straight_non_intersection
  ✅ U-turn intent:
        - Prio 1: tags
        - Prio 2: LANE_CONNECTOR.turn_type_fid == UTURN (map semantics)
        - Prio 3: geometry fallback
  ✅ Roundabout intent:
        - Prio 1: tags include ROUNDABOUT or PICKUP_DROPOFF / PICK_UP_DROP_OFF / PUDO
        - Prio 2: map PUDO / EXTENDED_PUDO with pudo_type == PICK_UP_DROP_OFF
        - Prio 3: geometry fallback (loop-like, high curvature, small net heading)

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python precompute_emoe_labels_anchors_strict_lane_following.py \
    --split mini \
    --output_dir ./emoe_precomputed_mini \
    --Ka 24 \
    --max_scenarios 20000 \
    --num_workers 8
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
# (NOTE: min-distance "others" gates are REMOVED to reduce "others" and focus on intent.)
# --------------------------------------------------------------------------------------
NET_STRAIGHT_MAX_AT_INTERSECTION = math.radians(12.0)   # |Δθ| <= 12° -> straight at intersection
NET_TURN_MIN_AT_INTERSECTION = math.radians(35.0)       # |Δθ| >= 35° -> turn candidate
NET_TURN_MAX_AT_INTERSECTION = math.radians(165.0)      # avoid near-UTURN confusion

UTURN_CENTER = math.pi
UTURN_MARGIN = math.radians(35.0)

# Non-intersection straightness gate (original)
NONINT_STRAIGHT_NET_MAX = math.radians(15.0)
NONINT_STRAIGHT_TOTAL_MAX = math.radians(25.0)

# Mild-curvature guard bands (used with lane-following alignment)
NONINT_LANEFOLLOW_NET_MAX = math.radians(25.0)  # allow slightly more net heading if lane-following is strong

# Roundabout geometry fallback (prio 3)
ROUNDABOUT_NET_MAX = math.radians(60.0)         # end heading roughly similar to start
ROUNDABOUT_TOTAL_MIN = math.radians(160.0)      # significant accumulated curvature
ROUNDABOUT_LOOP_RATIO_MIN = 1.6                 # path_len / straight_dist should be high (loopiness)


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
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
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


def path_length(xs: np.ndarray, ys: np.ndarray) -> float:
    """Approx path length along trajectory."""
    if len(xs) < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


# --------------------------------------------------------------------------------------
# Stage 1: tag/string-based priority mapping
# --------------------------------------------------------------------------------------
def _upper(x: Any) -> str:
    return str(x).upper() if x is not None else ""


def get_scenario_tags_if_available(scenario) -> List[str]:
    """Best-effort tag extraction. devkit versions differ."""
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

    # de-dup preserving order
    seen = set()
    out = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _tag_has_any(tags: List[str], needles: List[str]) -> bool:
    tset = set(_upper(t) for t in tags)
    for n in needles:
        nu = _upper(n)
        if any(nu in t for t in tset):
            return True
    return False


def stage1_from_tags_and_type(scenario_type: str, tags: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (class_id, stage_name) if decided, else (None, None).

    Prio-1 intent:
      - Roundabout: ROUNDABOUT tags OR pickup_dropoff-ish tags
      - U-turn: UTURN tags
      - Direct right-turn tags
    """
    st = _upper(scenario_type)
    tset = set(tags)

    # Roundabout (prio 1): explicit OR pickup_dropoff tags
    if (
        ("ROUNDABOUT" in st)
        or any("ROUNDABOUT" in _upper(t) for t in tset)
        or _tag_has_any(tags, ["PICKUP_DROPOFF", "PICK_UP_DROP_OFF", "PUDO", "PUDO_PICKUP_DROPOFF"])
    ):
        return 4, "stage1_tags_roundabout"

    # U-turn (prio 1)
    if ("UTURN" in st) or ("U_TURN" in st) or any(("UTURN" in _upper(t) or "U_TURN" in _upper(t)) for t in tset):
        return 5, "stage1_tags_uturn"

    # Direct: right turn tags
    if (
        ("STARTING_RIGHT" in st and "TURN" in st)
        or ("RIGHT_TURN" in st)
        or any(("STARTING_RIGHT" in _upper(t) and "TURN" in _upper(t)) for t in tset)
        or any(("RIGHT" in _upper(t) and "TURN" in _upper(t) and "STARTING" in _upper(t)) for t in tset)
    ):
        return 2, "stage1_tags_right_turn"

    return None, None


def tag_intersection_hint(scenario_type: str, tags: List[str]) -> bool:
    """Intersection context hint (does NOT force intersection class; traversal must be true)."""
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
    if any(any(k in _upper(t) for k in keys) for t in tset):
        return True
    return False


# --------------------------------------------------------------------------------------
# Map layer cache (per map_api instance)
# - All map layers are obtained from map_api._get_vector_map_layer(SemanticMapLayer.*)
# --------------------------------------------------------------------------------------
class MapLayerCache:
    """
    Cache vector layers + spatial indexes + geometries for faster per-scenario queries.
    """
    def __init__(self):
        self._cache: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _get_vector_map_layer(map_api, layer: SemanticMapLayer):
        # This is where INTERSECTION / LANE_CONNECTOR / LANE / PUDO come from (maps).
        try:
            return map_api._get_vector_map_layer(layer)  # type: ignore[attr-defined]
        except Exception:
            return None

    @staticmethod
    def _safe_sindex(gdf):
        try:
            return gdf.sindex
        except Exception:
            return None

    def get(self, map_api) -> Dict[str, Any]:
        key = id(map_api)
        if key in self._cache:
            return self._cache[key]

        inter_gdf = self._get_vector_map_layer(map_api, SemanticMapLayer.INTERSECTION)
        conn_gdf = self._get_vector_map_layer(map_api, SemanticMapLayer.LANE_CONNECTOR)
        lane_gdf = self._get_vector_map_layer(map_api, SemanticMapLayer.LANE)

        pudo_gdf = self._get_vector_map_layer(map_api, SemanticMapLayer.PUDO)
        ext_pudo_gdf = self._get_vector_map_layer(map_api, SemanticMapLayer.EXTENDED_PUDO)

        # Intersection
        inter_sindex = self._safe_sindex(inter_gdf) if inter_gdf is not None and len(inter_gdf) > 0 else None
        inter_geoms = list(inter_gdf.geometry.values) if inter_gdf is not None and len(inter_gdf) > 0 else []

        # Lane connectors + turn_type_fid
        conn_sindex = self._safe_sindex(conn_gdf) if conn_gdf is not None and len(conn_gdf) > 0 else None
        conn_geoms = list(conn_gdf.geometry.values) if conn_gdf is not None and len(conn_gdf) > 0 else []
        conn_turn_col = ""
        conn_turn_vals = None
        if conn_gdf is not None and len(conn_gdf) > 0:
            for c in ["turn_type_fid", "lane_connector_type_fid", "turn_type", "lane_connector_type"]:
                if c in conn_gdf.columns:
                    conn_turn_col = c
                    break
            conn_turn_vals = np.asarray(conn_gdf[conn_turn_col].values) if conn_turn_col else None

        # Lanes
        lane_sindex = self._safe_sindex(lane_gdf) if lane_gdf is not None and len(lane_gdf) > 0 else None
        lane_geoms = list(lane_gdf.geometry.values) if lane_gdf is not None and len(lane_gdf) > 0 else []

        # PUDO + type
        def _pudo_pack(gdf):
            if gdf is None or len(gdf) == 0:
                return None, [], None, "", None
            sidx = self._safe_sindex(gdf)
            geoms = list(gdf.geometry.values)
            type_col = ""
            for c in ["pudo_type_fid", "pudo_type", "type_fid", "type"]:
                if c in gdf.columns:
                    type_col = c
                    break
            type_vals = np.asarray(gdf[type_col].values) if type_col else None
            return gdf, geoms, sidx, type_col, type_vals

        pudo_gdf, pudo_geoms, pudo_sindex, pudo_type_col, pudo_type_vals = _pudo_pack(pudo_gdf)
        ext_pudo_gdf, ext_pudo_geoms, ext_pudo_sindex, ext_pudo_type_col, ext_pudo_type_vals = _pudo_pack(ext_pudo_gdf)

        out = {
            "inter_gdf": inter_gdf,
            "inter_sindex": inter_sindex,
            "inter_geoms": inter_geoms,

            "conn_gdf": conn_gdf,
            "conn_sindex": conn_sindex,
            "conn_geoms": conn_geoms,
            "conn_turn_col": conn_turn_col,
            "conn_turn_vals": conn_turn_vals,

            "lane_gdf": lane_gdf,
            "lane_sindex": lane_sindex,
            "lane_geoms": lane_geoms,

            "pudo_gdf": pudo_gdf,
            "pudo_geoms": pudo_geoms,
            "pudo_sindex": pudo_sindex,
            "pudo_type_col": pudo_type_col,
            "pudo_type_vals": pudo_type_vals,

            "ext_pudo_gdf": ext_pudo_gdf,
            "ext_pudo_geoms": ext_pudo_geoms,
            "ext_pudo_sindex": ext_pudo_sindex,
            "ext_pudo_type_col": ext_pudo_type_col,
            "ext_pudo_type_vals": ext_pudo_type_vals,
        }
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------------------
# STRICT intersection traversal from INTERSECTION polygons
# --------------------------------------------------------------------------------------
def intersection_traversal_from_layer(
    cache: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    intersection_tol_m: float,
    min_hits: int,
) -> Tuple[bool, float, int]:
    """
    Returns:
      traversed_intersection, min_dist_m, hits

    "Traversed" means ego points were inside or within tol of intersection polygons,
    at least `min_hits` times.
    """
    inter_geoms = cache.get("inter_geoms", [])
    if not inter_geoms:
        return False, float("inf"), 0

    inter_sindex = cache.get("inter_sindex", None)

    tol = float(intersection_tol_m)
    min_dist = float("inf")
    hits = 0

    for i in range(0, len(xs), max(1, sample_step)):
        p = Point(float(xs[i]), float(ys[i]))

        if inter_sindex is not None:
            bbox = (p.x - tol, p.y - tol, p.x + tol, p.y + tol)
            cand_idx = list(inter_sindex.intersection(bbox))
            if not cand_idx:
                continue
            geoms = [inter_geoms[j] for j in cand_idx if j < len(inter_geoms)]
        else:
            geoms = inter_geoms

        for g in geoms:
            if g is None:
                continue
            try:
                if g.contains(p):
                    hits += 1
                    min_dist = 0.0
                    break
                d = float(g.distance(p))
                min_dist = min(min_dist, d)
                if d <= tol:
                    hits += 1
                    break
            except Exception:
                continue

        if hits >= int(min_hits) and min_dist <= tol:
            return True, float(min_dist), int(hits)

    traversed = (hits >= int(min_hits)) and (min_dist <= tol)
    return bool(traversed), float(min_dist), int(hits)


# --------------------------------------------------------------------------------------
# Connector traversal by ego map-matching to connectors (strict)
# - This is where we use LANE_CONNECTOR from maps, and turn_type_fid from the layer.
# --------------------------------------------------------------------------------------
def _turn_type_to_label(v) -> str:
    """
    nuPlan LaneConnectorType (maps_datatypes.py):
      STRAIGHT=0, LEFT=1, RIGHT=2, UTURN=3, UNKNOWN=4
    """
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


def approx_geom_heading(geom, p: Point) -> Optional[float]:
    """Approximate tangent direction near p on a LineString-like geometry."""
    try:
        s = geom.project(p)
        s2 = min(s + 1.0, geom.length)
        p1 = geom.interpolate(s)
        p2 = geom.interpolate(s2)
        dx = float(p2.x - p1.x)
        dy = float(p2.y - p1.y)
        if dx * dx + dy * dy < 1e-6:
            return None
        return math.atan2(dy, dx)
    except Exception:
        return None


def connector_traversal_mapmatch(
    cache: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    hs: np.ndarray,
    sample_step: int,
    match_radius_m: float,
    min_hits: int,
    heading_gate_deg: float,
) -> Dict[str, Any]:
    """Strict connector traversal via map-match."""
    conn_geoms = cache.get("conn_geoms", [])
    if not conn_geoms:
        return {
            "traversed_connector": False,
            "hits": 0,
            "matched_samples": 0,
            "turn_counts": {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0, "UTURN": 0, "UNKNOWN": 0},
            "best_type": "NONE",
            "best_ratio": 0.0,
            "turn_type_col": "",
        }

    conn_sindex = cache.get("conn_sindex", None)
    turn_col = cache.get("conn_turn_col", "")
    turn_vals = cache.get("conn_turn_vals", None)

    r = float(match_radius_m)
    heading_gate = float(heading_gate_deg)

    counts = Counter()
    hits = 0
    matched = 0

    for i in range(0, len(xs), max(1, sample_step)):
        p = Point(float(xs[i]), float(ys[i]))
        ego_h = float(hs[i]) if i < len(hs) else float(hs[-1])

        if conn_sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(conn_sindex.intersection(bbox))
        else:
            cand_idx = list(range(len(conn_geoms)))

        if not cand_idx:
            continue

        best_j = None
        best_d = r

        for j in cand_idx:
            if j >= len(conn_geoms):
                continue
            g = conn_geoms[j]
            if g is None:
                continue
            try:
                d = float(g.distance(p))
            except Exception:
                continue
            if d <= best_d:
                if heading_gate > 0:
                    gh = approx_geom_heading(g, p)
                    if gh is not None:
                        dh = abs(wrap_to_pi(gh - ego_h))
                        if math.degrees(dh) > heading_gate:
                            continue
                best_d = d
                best_j = j

        if best_j is None:
            continue

        matched += 1
        hits += 1

        if turn_vals is not None and best_j < len(turn_vals):
            counts[_turn_type_to_label(turn_vals[best_j])] += 1
        else:
            counts["UNKNOWN"] += 1

    best_type = "NONE"
    best_count = 0
    for k in ["UTURN", "LEFT", "RIGHT", "STRAIGHT", "UNKNOWN"]:
        if counts[k] > best_count:
            best_count = counts[k]
            best_type = k

    denom = sum(counts[k] for k in ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN"])
    best_ratio = (best_count / max(1, denom))
    traversed = (hits >= int(min_hits))

    return {
        "traversed_connector": bool(traversed),
        "hits": int(hits),
        "matched_samples": int(matched),
        "turn_counts": {
            "LEFT": int(counts["LEFT"]),
            "RIGHT": int(counts["RIGHT"]),
            "STRAIGHT": int(counts["STRAIGHT"]),
            "UTURN": int(counts["UTURN"]),
            "UNKNOWN": int(counts["UNKNOWN"]),
        },
        "best_type": str(best_type),
        "best_ratio": float(best_ratio),
        "turn_type_col": str(turn_col),
    }


# --------------------------------------------------------------------------------------
# PUDO traversal (map semantics for roundabout prio 2)
# --------------------------------------------------------------------------------------
def _pudo_type_to_label(v) -> str:
    """
    nuPlan PudoType (maps_datatypes.py):
      PICK_UP_DROP_OFF=0, PICK_UP_ONLY=1, DROP_OFF_ONLY=2, UNKNOWN=3
    """
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "UNKNOWN"
        iv = int(v)
    except Exception:
        return "UNKNOWN"
    if iv == 0:
        return "PICK_UP_DROP_OFF"
    if iv == 1:
        return "PICK_UP_ONLY"
    if iv == 2:
        return "DROP_OFF_ONLY"
    return "UNKNOWN"


def _pudo_traversal_single_layer(
    geoms: List[Any],
    sindex,
    type_vals: Optional[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    tol_m: float,
    min_hits: int,
) -> Dict[str, Any]:
    if not geoms:
        return {"traversed": False, "hits": 0, "min_dist": float("inf"),
                "type_counts": {}, "best_type": "NONE", "best_ratio": 0.0}

    tol = float(tol_m)
    hits = 0
    min_dist = float("inf")
    counts = Counter()

    for i in range(0, len(xs), max(1, sample_step)):
        p = Point(float(xs[i]), float(ys[i]))

        if sindex is not None:
            bbox = (p.x - tol, p.y - tol, p.x + tol, p.y + tol)
            cand_idx = list(sindex.intersection(bbox))
            if not cand_idx:
                continue
        else:
            cand_idx = list(range(len(geoms)))

        touched = False
        touched_idx = None
        for j in cand_idx:
            if j >= len(geoms):
                continue
            g = geoms[j]
            if g is None:
                continue
            try:
                if g.contains(p):
                    min_dist = 0.0
                    touched = True
                    touched_idx = j
                    break
                d = float(g.distance(p))
                min_dist = min(min_dist, d)
                if d <= tol:
                    touched = True
                    touched_idx = j
                    break
            except Exception:
                continue

        if touched:
            hits += 1
            if type_vals is not None and touched_idx is not None and touched_idx < len(type_vals):
                counts[_pudo_type_to_label(type_vals[touched_idx])] += 1
            else:
                counts["UNKNOWN"] += 1

        if hits >= int(min_hits) and min_dist <= tol:
            break

    best_type = "NONE"
    best_count = 0
    for k in ["PICK_UP_DROP_OFF", "PICK_UP_ONLY", "DROP_OFF_ONLY", "UNKNOWN"]:
        if counts[k] > best_count:
            best_count = counts[k]
            best_type = k

    denom = sum(counts[k] for k in ["PICK_UP_DROP_OFF", "PICK_UP_ONLY", "DROP_OFF_ONLY", "UNKNOWN"])
    best_ratio = (best_count / max(1, denom)) if denom > 0 else 0.0
    traversed = (hits >= int(min_hits)) and (min_dist <= tol)

    return {
        "traversed": bool(traversed),
        "hits": int(hits),
        "min_dist": float(min_dist),
        "type_counts": {k: int(counts[k]) for k in ["PICK_UP_DROP_OFF", "PICK_UP_ONLY", "DROP_OFF_ONLY", "UNKNOWN"]},
        "best_type": str(best_type),
        "best_ratio": float(best_ratio),
    }


def pudo_traversal_from_maps(
    cache: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    sample_step: int,
    tol_m: float,
    min_hits: int,
) -> Dict[str, Any]:
    """
    Checks both PUDO and EXTENDED_PUDO layers.
    Roundabout prio-2 uses ONLY PICK_UP_DROP_OFF.
    """
    p = _pudo_traversal_single_layer(
        cache.get("pudo_geoms", []),
        cache.get("pudo_sindex", None),
        cache.get("pudo_type_vals", None),
        xs, ys,
        sample_step=sample_step,
        tol_m=tol_m,
        min_hits=min_hits,
    )
    e = _pudo_traversal_single_layer(
        cache.get("ext_pudo_geoms", []),
        cache.get("ext_pudo_sindex", None),
        cache.get("ext_pudo_type_vals", None),
        xs, ys,
        sample_step=sample_step,
        tol_m=tol_m,
        min_hits=min_hits,
    )

    # Merge: prefer the one with more hits; if equal, prefer smaller min_dist
    chosen = p
    chosen_layer = "PUDO"
    if (e.get("hits", 0) > p.get("hits", 0)) or (e.get("hits", 0) == p.get("hits", 0) and e.get("min_dist", 1e9) < p.get("min_dist", 1e9)):
        chosen = e
        chosen_layer = "EXTENDED_PUDO"

    out = dict(chosen)
    out["layer"] = chosen_layer
    out["pudo_type_col"] = cache.get("pudo_type_col", "")
    out["ext_pudo_type_col"] = cache.get("ext_pudo_type_col", "")
    return out


# --------------------------------------------------------------------------------------
# Lane-following alignment check
# --------------------------------------------------------------------------------------
def lane_following_alignment(
    cache: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    hs: np.ndarray,
    *,
    use_connectors: bool,
    sample_step: int,
    match_radius_m: float,
    heading_gate_deg: float,
    min_hits: int,
) -> Dict[str, Any]:
    """Compute how well ego heading follows tangent of lane/lane_connector geometry."""
    layer_name = "conn" if use_connectors else "lane"
    if use_connectors:
        geoms = cache.get("conn_geoms", [])
        sindex = cache.get("conn_sindex", None)
    else:
        geoms = cache.get("lane_geoms", [])
        sindex = cache.get("lane_sindex", None)

    if not geoms:
        return {
            "layer": layer_name,
            "hits": 0,
            "samples": 0,
            "hit_ratio": 0.0,
            "median_err_deg": float("inf"),
            "mean_err_deg": float("inf"),
        }

    r = float(match_radius_m)
    heading_gate = float(heading_gate_deg)

    errs = []
    hits = 0
    samples = 0

    for i in range(0, len(xs), max(1, sample_step)):
        samples += 1
        p = Point(float(xs[i]), float(ys[i]))
        ego_h = float(hs[i]) if i < len(hs) else float(hs[-1])

        if sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(sindex.intersection(bbox))
        else:
            cand_idx = list(range(len(geoms)))

        if not cand_idx:
            continue

        best_j = None
        best_d = r
        best_gh = None

        for j in cand_idx:
            if j >= len(geoms):
                continue
            g = geoms[j]
            if g is None:
                continue
            try:
                d = float(g.distance(p))
            except Exception:
                continue
            if d <= best_d:
                gh = approx_geom_heading(g, p)
                if gh is None:
                    continue
                if heading_gate > 0:
                    dh0 = abs(wrap_to_pi(gh - ego_h))
                    if math.degrees(dh0) > heading_gate:
                        continue
                best_d = d
                best_j = j
                best_gh = gh

        if best_j is None or best_gh is None:
            continue

        hits += 1
        err = abs(wrap_to_pi(best_gh - ego_h))
        errs.append(math.degrees(err))

    if hits < int(min_hits) or len(errs) == 0:
        return {
            "layer": layer_name,
            "hits": int(hits),
            "samples": int(samples),
            "hit_ratio": float(hits / max(1, samples)),
            "median_err_deg": float("inf"),
            "mean_err_deg": float("inf"),
        }

    errs = np.asarray(errs, dtype=np.float64)
    return {
        "layer": layer_name,
        "hits": int(hits),
        "samples": int(samples),
        "hit_ratio": float(hits / max(1, samples)),
        "median_err_deg": float(np.median(errs)),
        "mean_err_deg": float(np.mean(errs)),
    }


# --------------------------------------------------------------------------------------
# Stage 2+3: STRICT intersection context + ego-lane-aware intent classification
# --------------------------------------------------------------------------------------
def classify_strict_intersection_logic(
    scenario,
    xs: np.ndarray,
    ys: np.ndarray,
    headings: np.ndarray,
    tags: List[str],
    cache: Dict[str, Any],
    *,
    map_sample_step: int,
    intersection_tol_m: float,
    intersection_min_hits: int,
    connector_sample_step: int,
    connector_match_radius_m: float,
    connector_min_hits: int,
    connector_heading_gate_deg: float,
    connector_verify_min_ratio: float,

    # lane-following knobs
    lane_sample_step: int,
    lane_match_radius_m: float,
    lane_heading_gate_deg: float,
    lane_min_hits: int,
    lane_following_ok_deg: float,

    # PUDO knobs
    pudo_sample_step: int,
    pudo_tol_m: float,
    pudo_min_hits: int,
    pudo_verify_min_ratio: float,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    STRICT traversal + intent classification.

    U-turn:
      prio2 = connector turn_type_fid == UTURN
      prio3 = geometry (~pi)

    Roundabout:
      prio2 = PUDO / EXTENDED_PUDO with PudoType.PICK_UP_DROP_OFF
      prio3 = geometry loopiness
    """
    T = len(xs)
    if T < 3:
        return 6, "stage3_short", {"T": T}

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    dist = float(math.hypot(dx, dy))
    pl = path_length(xs, ys)

    h0 = float(headings[0])
    hT = float(headings[-1])
    delta_heading = wrap_to_pi(hT - h0)
    abs_dh = abs(delta_heading)

    dh = np.diff(headings)
    dh = np.vectorize(wrap_to_pi)(dh)
    total_abs = float(np.sum(np.abs(dh)))

    has_intersection_tag = tag_intersection_hint(getattr(scenario, "scenario_type", ""), tags)

    # STRICT intersection traversal by polygons
    traversed_poly, inter_min_dist, inter_hits = intersection_traversal_from_layer(
        cache, xs, ys,
        sample_step=map_sample_step,
        intersection_tol_m=intersection_tol_m,
        min_hits=intersection_min_hits,
    )

    # STRICT connector traversal by map-match (and turn_type_fid)
    conn_mm = connector_traversal_mapmatch(
        cache, xs, ys, headings,
        sample_step=connector_sample_step,
        match_radius_m=connector_match_radius_m,
        min_hits=connector_min_hits,
        heading_gate_deg=connector_heading_gate_deg,
    )
    traversed_connector = bool(conn_mm.get("traversed_connector", False))
    traversed_intersection = bool(traversed_poly or traversed_connector)

    # Lane-following alignment
    lf = lane_following_alignment(
        cache, xs, ys, headings,
        use_connectors=bool(traversed_intersection),
        sample_step=lane_sample_step,
        match_radius_m=lane_match_radius_m,
        heading_gate_deg=lane_heading_gate_deg,
        min_hits=lane_min_hits,
    )
    lane_following_ok = (
        np.isfinite(lf.get("median_err_deg", float("inf")))
        and lf.get("median_err_deg", float("inf")) <= float(lane_following_ok_deg)
    )

    # PUDO traversal (map semantics for roundabout prio 2)
    pudo_mm = pudo_traversal_from_maps(
        cache, xs, ys,
        sample_step=pudo_sample_step,
        tol_m=pudo_tol_m,
        min_hits=pudo_min_hits,
    )

    debug: Dict[str, Any] = {
        "dist": dist,
        "path_len": pl,
        "path_len_over_dist": float(pl / max(1e-3, dist)),
        "delta_heading_deg": float(math.degrees(delta_heading)),
        "abs_delta_heading_deg": float(math.degrees(abs_dh)),
        "total_abs_heading_deg": float(math.degrees(total_abs)),

        "has_intersection_tag": bool(has_intersection_tag),
        "traversed_intersection_polygon": bool(traversed_poly),
        "intersection_min_dist_m": float(inter_min_dist),
        "intersection_hits": int(inter_hits),

        "traversed_lane_connector": bool(traversed_connector),
        "connector_mapmatch": conn_mm,

        "lane_following": lf,
        "lane_following_ok": bool(lane_following_ok),
        "lane_following_ok_deg": float(lane_following_ok_deg),

        "pudo_mapmatch": pudo_mm,
    }

    # ------------------------------------------------------------------------------
    # U-TURN: prio 2 (map semantics via connector turn_type_fid)
    # ------------------------------------------------------------------------------
    if traversed_connector:
        bt = str(conn_mm.get("best_type", "NONE"))
        br = float(conn_mm.get("best_ratio", 0.0))
        tc = (conn_mm.get("turn_counts", {}) or {})
        uturn_hits = int(tc.get("UTURN", 0))

        # If map says UTURN confidently, classify U-turn (prio 2).
        if bt == "UTURN" and (br >= float(connector_verify_min_ratio) or uturn_hits >= 1):
            debug["uturn_reason"] = {
                "prio": 2,
                "source": "LANE_CONNECTOR.turn_type_fid",
                "best_type": bt,
                "best_ratio": br,
                "uturn_hits": uturn_hits,
                "turn_type_col": str(conn_mm.get("turn_type_col", "")),
            }
            return 5, "stage3_map_uturn", debug

    # ------------------------------------------------------------------------------
    # ROUNDABOUT: prio 2 (map semantics via PUDO / EXTENDED_PUDO)
    # - ONLY PICK_UP_DROP_OFF is treated as roundabout-like; pickup_only/dropoff_only are NOT.
    # ------------------------------------------------------------------------------
    if bool(pudo_mm.get("traversed", False)):
        p_bt = str(pudo_mm.get("best_type", "NONE"))
        p_br = float(pudo_mm.get("best_ratio", 0.0))
        if p_bt == "PICK_UP_DROP_OFF" and p_br >= float(pudo_verify_min_ratio):
            debug["roundabout_reason"] = {
                "prio": 2,
                "source": pudo_mm.get("layer", "PUDO"),
                "best_type": p_bt,
                "best_ratio": p_br,
                "type_counts": pudo_mm.get("type_counts", {}),
            }
            return 4, "stage3_map_pudo_roundabout", debug

    # ------------------------------------------------------------------------------
    # U-TURN: prio 3 (geometry fallback)
    # ------------------------------------------------------------------------------
    if abs(abs_dh - UTURN_CENTER) < UTURN_MARGIN:
        debug["uturn_reason"] = {
            "prio": 3,
            "source": "geometry",
            "abs_dh_deg": float(math.degrees(abs_dh)),
            "margin_deg": float(math.degrees(UTURN_MARGIN)),
        }
        return 5, "stage3_geometry_uturn", debug

    # ------------------------------------------------------------------------------
    # ROUNDABOUT: prio 3 (geometry fallback)
    # ------------------------------------------------------------------------------
    loopiness = float(pl / max(1e-3, dist))
    if (abs_dh <= ROUNDABOUT_NET_MAX) and (total_abs >= ROUNDABOUT_TOTAL_MIN) and (loopiness >= ROUNDABOUT_LOOP_RATIO_MIN) and lane_following_ok:
        debug["roundabout_reason"] = {
            "prio": 3,
            "source": "geometry",
            "abs_dh_deg": float(math.degrees(abs_dh)),
            "total_abs_deg": float(math.degrees(total_abs)),
            "loopiness": loopiness,
            "lane_following_ok": bool(lane_following_ok),
        }
        return 4, "stage3_geometry_roundabout", debug

    # ------------------------------------------------------------------------------
    # Directional logic (left/right/straight) — keep ego-lane-aware behavior
    # ------------------------------------------------------------------------------
    if traversed_intersection:
        # Straight through intersection
        if abs_dh <= NET_STRAIGHT_MAX_AT_INTERSECTION:
            if traversed_connector:
                bt = str(conn_mm.get("best_type", "NONE"))
                br = float(conn_mm.get("best_ratio", 0.0))
                debug["connector_verify_note"] = f"best={bt}, ratio={br:.2f}"
            return 1, "stage3_intersection_net_heading", debug

        # Turn at intersection by net heading
        if NET_TURN_MIN_AT_INTERSECTION <= abs_dh <= NET_TURN_MAX_AT_INTERSECTION:
            geom_cls = 0 if delta_heading > 0.0 else 2  # left if positive, else right

            if traversed_connector:
                bt = str(conn_mm.get("best_type", "NONE"))
                br = float(conn_mm.get("best_ratio", 0.0))
                if br >= float(connector_verify_min_ratio) and bt in ("LEFT", "RIGHT", "STRAIGHT", "UTURN"):
                    debug["connector_verify_note"] = f"best={bt}, ratio={br:.2f}"
                    if (geom_cls == 0 and bt == "RIGHT") or (geom_cls == 2 and bt == "LEFT"):
                        debug["connector_verify_conflict"] = True

            return geom_cls, "stage3_intersection_net_heading", debug

        # Intersection "gap" fix (12°–35°) => often curved straight
        gap_low = NET_STRAIGHT_MAX_AT_INTERSECTION
        gap_high = NET_TURN_MIN_AT_INTERSECTION
        if gap_low < abs_dh < gap_high:
            bt = str(conn_mm.get("best_type", "NONE"))
            br = float(conn_mm.get("best_ratio", 0.0))
            connector_says_straight = (traversed_connector and bt == "STRAIGHT" and br >= float(connector_verify_min_ratio))
            if lane_following_ok or connector_says_straight:
                debug["intersection_gap_promoted_to_straight"] = True
                debug["intersection_gap_reason"] = {
                    "lane_following_ok": bool(lane_following_ok),
                    "connector_says_straight": bool(connector_says_straight),
                    "conn_best_type": bt,
                    "conn_best_ratio": br,
                    "abs_dh_deg": float(math.degrees(abs_dh)),
                }
                return 1, "stage3_intersection_gap_lane_following", debug

        # If traversed intersection but doesn't fit: prefer intentful "straight_at_intersection" if lane-following is strong,
        # otherwise fallback to others.
        if lane_following_ok and abs_dh <= math.radians(90.0):
            debug["intersection_fallback_promoted_to_straight"] = True
            return 1, "stage3_intersection_fallback_lane_following", debug

        return 6, "stage3_intersection_fallback", debug

    # Non-intersection branch
    if abs_dh <= NONINT_STRAIGHT_NET_MAX and total_abs <= NONINT_STRAIGHT_TOTAL_MAX:
        return 3, "stage3_nonintersection_straight", debug

    # Curved lane-following outside intersection
    if abs_dh <= NONINT_LANEFOLLOW_NET_MAX and lane_following_ok:
        debug["nonintersection_promoted_to_straight_lane_following"] = True
        return 3, "stage3_nonintersection_lane_following", debug

    # If the vehicle barely moved, still give a sensible intent instead of "others"
    if dist < 0.5 and lane_following_ok:
        return 3, "stage3_nonintersection_stationary_lane_following", debug

    return 6, "stage3_nonintersection_other", debug


# --------------------------------------------------------------------------------------
# Combined classification pipeline
# --------------------------------------------------------------------------------------
def classify_emoe_for_scenario(
    scenario,
    cache: Dict[str, Any],
    *,
    map_sample_step: int,
    intersection_tol_m: float,
    intersection_min_hits: int,
    connector_sample_step: int,
    connector_match_radius_m: float,
    connector_min_hits: int,
    connector_heading_gate_deg: float,
    connector_verify_min_ratio: float,

    lane_sample_step: int,
    lane_match_radius_m: float,
    lane_heading_gate_deg: float,
    lane_min_hits: int,
    lane_following_ok_deg: float,

    pudo_sample_step: int,
    pudo_tol_m: float,
    pudo_min_hits: int,
    pudo_verify_min_ratio: float,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    Stage 1: tags/strings priority (roundabout, u-turn, direct right-turn tags)
    Stage 2+3: STRICT traversal + map semantics + geometry + lane-following fixes
    """
    scenario_type = getattr(scenario, "scenario_type", "")
    tags = get_scenario_tags_if_available(scenario)

    emoe_id, stage = stage1_from_tags_and_type(scenario_type, tags)
    if emoe_id is not None:
        return int(emoe_id), str(stage), {"scenario_type": scenario_type, "tags": tags}

    xs, ys, hs = compute_ego_xyh(scenario)
    emoe_id, stage, debug = classify_strict_intersection_logic(
        scenario, xs, ys, hs, tags, cache,
        map_sample_step=map_sample_step,
        intersection_tol_m=intersection_tol_m,
        intersection_min_hits=intersection_min_hits,
        connector_sample_step=connector_sample_step,
        connector_match_radius_m=connector_match_radius_m,
        connector_min_hits=connector_min_hits,
        connector_heading_gate_deg=connector_heading_gate_deg,
        connector_verify_min_ratio=connector_verify_min_ratio,

        lane_sample_step=lane_sample_step,
        lane_match_radius_m=lane_match_radius_m,
        lane_heading_gate_deg=lane_heading_gate_deg,
        lane_min_hits=lane_min_hits,
        lane_following_ok_deg=lane_following_ok_deg,

        pudo_sample_step=pudo_sample_step,
        pudo_tol_m=pudo_tol_m,
        pudo_min_hits=pudo_min_hits,
        pudo_verify_min_ratio=pudo_verify_min_ratio,
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

    # STRICT intersection traversal knobs
    parser.add_argument("--map_sample_step", type=int, default=3, help="Subsample step for INTERSECTION traversal checks")
    parser.add_argument("--intersection_tol_m", type=float, default=2.5, help="Distance tolerance to intersection polygon [m]")
    parser.add_argument("--intersection_min_hits", type=int, default=3, help="Min hits to call 'traversed intersection polygon'")

    # STRICT connector traversal knobs
    parser.add_argument("--connector_sample_step", type=int, default=2, help="Subsample step for connector map-match")
    parser.add_argument("--connector_match_radius_m", type=float, default=3.0, help="Max distance to match a connector [m]")
    parser.add_argument("--connector_min_hits", type=int, default=2, help="Min matched samples to call 'traversed connector'")
    parser.add_argument("--connector_heading_gate_deg", type=float, default=70.0,
                        help="Heading gate for connector match (deg). <=0 disables.")
    parser.add_argument("--connector_verify_min_ratio", type=float, default=0.55,
                        help="If traversed connector and best_ratio>=this, treat as confident verification.")

    # Lane-following knobs
    parser.add_argument("--lane_sample_step", type=int, default=3,
                        help="Subsample step for lane-following alignment check")
    parser.add_argument("--lane_match_radius_m", type=float, default=3.5,
                        help="Max distance to match lane/lane_connector for lane-following [m]")
    parser.add_argument("--lane_heading_gate_deg", type=float, default=120.0,
                        help="Reject lane tangent if |ego-heading - tangent| > this (deg). <=0 disables.")
    parser.add_argument("--lane_min_hits", type=int, default=5,
                        help="Min successful tangent matches to trust lane-following metric")
    parser.add_argument("--lane_following_ok_deg", type=float, default=15.0,
                        help="If median heading error (deg) <= this, consider ego is lane-following")

    # PUDO knobs (roundabout prio 2)
    parser.add_argument("--pudo_sample_step", type=int, default=3,
                        help="Subsample step for PUDO traversal check")
    parser.add_argument("--pudo_tol_m", type=float, default=3.0,
                        help="Distance tolerance to PUDO polygons [m]")
    parser.add_argument("--pudo_min_hits", type=int, default=2,
                        help="Min hits to call 'traversed PUDO'")
    parser.add_argument("--pudo_verify_min_ratio", type=float, default=0.55,
                        help="If best_ratio>=this on PUDO type, trust as prio-2 roundabout signal.")

    # Anchors
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
    print(f"[INFO] intersection_tol_m={args.intersection_tol_m}, min_hits={args.intersection_min_hits}, step={args.map_sample_step}")
    print(f"[INFO] connector_match_radius_m={args.connector_match_radius_m}, min_hits={args.connector_min_hits}, step={args.connector_sample_step}, heading_gate={args.connector_heading_gate_deg}, verify_ratio={args.connector_verify_min_ratio}")
    print(f"[INFO] lane_following: radius={args.lane_match_radius_m}, step={args.lane_sample_step}, min_hits={args.lane_min_hits}, ok_deg={args.lane_following_ok_deg}")
    print(f"[INFO] pudo: tol_m={args.pudo_tol_m}, min_hits={args.pudo_min_hits}, step={args.pudo_sample_step}, verify_ratio={args.pudo_verify_min_ratio}")
    print(f"[INFO] min_travel_distance(for anchors)={args.min_travel_distance}")

    print("[INFO] Loading scenarios...")
    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)

    if args.max_scenarios is not None and args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    print(f"[INFO] Loaded {len(scenarios)} scenarios (after hard cap).")

    endpoints_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    class_counts = Counter()
    stage_counts = Counter()

    map_cache = MapLayerCache()

    f_labels = labels_path.open("w")

    try:
        for scenario in tqdm(scenarios, total=len(scenarios), desc="Classifying + collecting endpoints"):
            token = scenario.token

            map_api = getattr(scenario, "map_api", None)
            cache = map_cache.get(map_api) if map_api is not None else {}

            emoe_id, stage, debug = classify_emoe_for_scenario(
                scenario,
                cache,
                map_sample_step=args.map_sample_step,
                intersection_tol_m=args.intersection_tol_m,
                intersection_min_hits=args.intersection_min_hits,
                connector_sample_step=args.connector_sample_step,
                connector_match_radius_m=args.connector_match_radius_m,
                connector_min_hits=args.connector_min_hits,
                connector_heading_gate_deg=args.connector_heading_gate_deg,
                connector_verify_min_ratio=args.connector_verify_min_ratio,

                lane_sample_step=args.lane_sample_step,
                lane_match_radius_m=args.lane_match_radius_m,
                lane_heading_gate_deg=args.lane_heading_gate_deg,
                lane_min_hits=args.lane_min_hits,
                lane_following_ok_deg=args.lane_following_ok_deg,

                pudo_sample_step=args.pudo_sample_step,
                pudo_tol_m=args.pudo_tol_m,
                pudo_min_hits=args.pudo_min_hits,
                pudo_verify_min_ratio=args.pudo_verify_min_ratio,
            )
            class_counts[emoe_id] += 1
            stage_counts[stage] += 1

            xs, ys, hs = compute_ego_xyh(scenario)
            dist = float(math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))) if len(xs) >= 2 else 0.0

            if dist >= float(args.min_travel_distance):
                endpoint_xy = ego_endpoint_in_ego_frame(xs, ys, hs)
                endpoints_by_class[emoe_id].append(endpoint_xy)

            conn_mm = debug.get("connector_mapmatch", {}) or {}
            lane_follow = debug.get("lane_following", {}) or {}
            pudo_follow = debug.get("pudo_mapmatch", {}) or {}

            record = {
                "token": str(token),
                "emoe_class_id": int(emoe_id),
                "emoe_class_name": EMOE_SCENE_TYPES[int(emoe_id)],
                "scenario_type": getattr(scenario, "scenario_type", ""),
                "stage": stage,
                "travel_distance_m": float(dist),
                "debug": {
                    "dist": debug.get("dist", None),
                    "path_len": debug.get("path_len", None),
                    "path_len_over_dist": debug.get("path_len_over_dist", None),
                    "delta_heading_deg": debug.get("delta_heading_deg", None),
                    "abs_delta_heading_deg": debug.get("abs_delta_heading_deg", None),
                    "total_abs_heading_deg": debug.get("total_abs_heading_deg", None),

                    "has_intersection_tag": debug.get("has_intersection_tag", None),
                    "traversed_intersection_polygon": debug.get("traversed_intersection_polygon", None),
                    "intersection_min_dist_m": debug.get("intersection_min_dist_m", None),
                    "intersection_hits": debug.get("intersection_hits", None),

                    "traversed_lane_connector": debug.get("traversed_lane_connector", None),
                    "connector_turn_type_col": conn_mm.get("turn_type_col", None),
                    "connector_best_type": conn_mm.get("best_type", None),
                    "connector_best_ratio": conn_mm.get("best_ratio", None),
                    "connector_turn_counts": conn_mm.get("turn_counts", None),

                    "lane_following_layer": lane_follow.get("layer", None),
                    "lane_following_hits": lane_follow.get("hits", None),
                    "lane_following_hit_ratio": lane_follow.get("hit_ratio", None),
                    "lane_following_median_err_deg": lane_follow.get("median_err_deg", None),
                    "lane_following_mean_err_deg": lane_follow.get("mean_err_deg", None),
                    "lane_following_ok": debug.get("lane_following_ok", None),
                    "lane_following_ok_deg": debug.get("lane_following_ok_deg", None),

                    "pudo_layer": pudo_follow.get("layer", None),
                    "pudo_traversed": pudo_follow.get("traversed", None),
                    "pudo_hits": pudo_follow.get("hits", None),
                    "pudo_min_dist": pudo_follow.get("min_dist", None),
                    "pudo_best_type": pudo_follow.get("best_type", None),
                    "pudo_best_ratio": pudo_follow.get("best_ratio", None),
                    "pudo_type_counts": pudo_follow.get("type_counts", None),

                    "uturn_reason": debug.get("uturn_reason", None),
                    "roundabout_reason": debug.get("roundabout_reason", None),

                    "intersection_gap_promoted_to_straight": debug.get("intersection_gap_promoted_to_straight", None),
                    "intersection_gap_reason": debug.get("intersection_gap_reason", None),
                    "nonintersection_promoted_to_straight_lane_following": debug.get("nonintersection_promoted_to_straight_lane_following", None),

                    "scenario_type": debug.get("scenario_type", None),
                    "tags": debug.get("tags", None),
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
        print(f"  {k:44s}: {v}")

    print("\n[INFO] Endpoint counts per class (for anchor clustering):")
    for c in range(7):
        print(f"  class {c} ({EMOE_SCENE_TYPES[c]:28s}): {len(endpoints_by_class[c])} endpoints")

    # KMeans per class -> anchors (endpoints only)
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
