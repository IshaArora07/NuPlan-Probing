#!/usr/bin/env python3
"""
Unsupervised (no rule-based) clustering of nuPlan scenarios using **interpretable continuous
map + trajectory features** so clusters become "intersection-y / turning-y / stationary-y".

What this script does:
  1) Loads nuPlan scenarios (mini / trainval / etc.)
  2) Computes interpretable features per scenario:
       - motion: travel distance, mean speed proxy, stationary ratio
       - heading: net heading change, total abs heading
       - map semantics (SemanticMapLayer):
           * intersection proximity / hit ratio using INTERSECTION polygons
           * connector turn-type vote ratios using LANE_CONNECTOR + turn_type_fid
       - endpoint in ego frame (x_end, y_end)
       - lightweight tag/type flags (intersection/right/roundabout/uturn/pudo)
  3) Standardizes features and clusters with MiniBatchKMeans
  4) Saves:
       - cluster_labels.jsonl   (token -> cluster_id + feature snapshot)
       - features.npy / tokens.txt
       - exemplars.json         (top-k closest-to-centroid tokens per cluster)
       - plots/                (cluster counts, PCA scatter)

Notes:
  - This is UNSUPERVISED: clusters do NOT have semantic names. You will label clusters later.
  - Uses SemanticMapLayer vector layers (no get_one_lane_or_lane_connector_from_point).
  - Designed to be fairly fast: uses GeoPandas spatial index + prepared geometries + map caching.

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python unsupervised_cluster_nuplan.py \
    --split mini \
    --out_dir ./unsup_clusters_mini \
    --max_scenarios 20000 \
    --num_clusters 40 \
    --map_sample_step 5 \
    --intersection_tol_m 12 \
    --connector_radius_m 5 \
    --exemplars_per_cluster 12
"""

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from shapely.geometry import Point
from shapely.prepared import prep


# -----------------------------
# Small utilities
# -----------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _upper(x: Any) -> str:
    return str(x).upper() if x is not None else ""


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, hs = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
        hs.append(float(ego.rear_axle.heading))
    return np.asarray(xs, np.float64), np.asarray(ys, np.float64), np.asarray(hs, np.float64)


def endpoint_in_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Tuple[float, float]:
    if len(xs) < 2:
        return 0.0, 0.0
    x0, y0 = xs[0], ys[0]
    xT, yT = xs[-1], ys[-1]
    dx, dy = (xT - x0), (yT - y0)
    th0 = hs[0]
    c = math.cos(-th0)
    s = math.sin(-th0)
    x_rel = c * dx - s * dy
    y_rel = s * dx + c * dy
    return float(x_rel), float(y_rel)


def safe_sindex(gdf):
    try:
        return gdf.sindex
    except Exception:
        return None


def get_tags_if_available(scenario) -> List[str]:
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


def contains_any(keys: List[str], s: str) -> bool:
    ss = _upper(s)
    return any(k in ss for k in keys)


# -----------------------------
# Map cache (SemanticMapLayer)
# -----------------------------
@dataclass
class IntersectionLayerCache:
    gdf: Any
    sindex: Any
    prepared_geoms: List[Any]  # prepared polygons


@dataclass
class ConnectorLayerCache:
    gdf: Any
    sindex: Any
    turn_col: str


class MapLayerCache:
    """
    Cache vector layers + indices per map_api (or per map name if available).
    This avoids rebuilding sindex/prepared geometry repeatedly.
    """
    def __init__(self):
        self._inter_cache: Dict[str, IntersectionLayerCache] = {}
        self._conn_cache: Dict[str, ConnectorLayerCache] = {}

    def _key(self, scenario) -> str:
        # Try stable key: map name if exists; else fallback to map_api id.
        mn = getattr(scenario, "map_name", None)
        if mn is None:
            mn = getattr(getattr(scenario, "map_api", None), "map_name", None)
        if mn:
            return str(mn)
        return f"map_api_{id(getattr(scenario, 'map_api', None))}"

    def _get_vector_layer(self, map_api, layer: SemanticMapLayer):
        try:
            return map_api._get_vector_map_layer(layer)  # type: ignore[attr-defined]
        except Exception:
            return None

    def get_intersection_cache(self, scenario) -> Optional[IntersectionLayerCache]:
        map_api = getattr(scenario, "map_api", None)
        if map_api is None:
            return None
        key = self._key(scenario)
        if key in self._inter_cache:
            return self._inter_cache[key]

        gdf = self._get_vector_layer(map_api, SemanticMapLayer.INTERSECTION)
        if gdf is None or len(gdf) == 0:
            self._inter_cache[key] = IntersectionLayerCache(gdf=None, sindex=None, prepared_geoms=[])
            return self._inter_cache[key]

        sindex = safe_sindex(gdf)
        # prepare geometries for faster contains/distance
        prepared = []
        for geom in gdf.geometry.values:
            if geom is None:
                prepared.append(None)
            else:
                try:
                    prepared.append(prep(geom))
                except Exception:
                    prepared.append(None)

        self._inter_cache[key] = IntersectionLayerCache(gdf=gdf, sindex=sindex, prepared_geoms=prepared)
        return self._inter_cache[key]

    def get_connector_cache(self, scenario) -> Optional[ConnectorLayerCache]:
        map_api = getattr(scenario, "map_api", None)
        if map_api is None:
            return None
        key = self._key(scenario)
        if key in self._conn_cache:
            return self._conn_cache[key]

        gdf = self._get_vector_layer(map_api, SemanticMapLayer.LANE_CONNECTOR)
        if gdf is None or len(gdf) == 0:
            self._conn_cache[key] = ConnectorLayerCache(gdf=None, sindex=None, turn_col="")
            return self._conn_cache[key]

        # robust turn type column detection
        col = ""
        for c in ["turn_type_fid", "lane_connector_type_fid", "turn_type", "lane_connector_type"]:
            if c in gdf.columns:
                col = c
                break

        sindex = safe_sindex(gdf)
        self._conn_cache[key] = ConnectorLayerCache(gdf=gdf, sindex=sindex, turn_col=col)
        return self._conn_cache[key]


# -----------------------------
# Interpretable map features
# -----------------------------
def intersection_features(
    inter_cache: IntersectionLayerCache,
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    tol_m: float,
) -> Tuple[float, float, float]:
    """
    Returns:
      hit_ratio: hits / samples
      min_dist_m: minimum distance to any intersection polygon
      inside_ratio: fraction of samples strictly inside some intersection polygon
    """
    if inter_cache is None or inter_cache.gdf is None or len(inter_cache.prepared_geoms) == 0:
        return 0.0, float("inf"), 0.0

    gdf = inter_cache.gdf
    sindex = inter_cache.sindex
    prepared = inter_cache.prepared_geoms

    tol = float(tol_m)
    hits = 0
    inside = 0
    total = 0
    min_dist = float("inf")

    for i in range(0, len(xs), max(1, sample_step)):
        total += 1
        p = Point(float(xs[i]), float(ys[i]))

        if sindex is not None:
            bbox = (p.x - tol, p.y - tol, p.x + tol, p.y + tol)
            cand_idx = list(sindex.intersection(bbox))
            cand = cand_idx
        else:
            cand = range(len(gdf))

        any_hit = False
        any_inside = False

        for j in cand:
            geom = gdf.geometry.values[j]
            if geom is None:
                continue
            # prepared contains
            pj = prepared[j]
            try:
                if pj is not None and pj.contains(p):
                    any_inside = True
                    any_hit = True
                    min_dist = 0.0
                    break
            except Exception:
                pass
            # distance check
            try:
                d = float(geom.distance(p))
            except Exception:
                continue
            if d < min_dist:
                min_dist = d
            if d <= tol:
                any_hit = True
                break

        if any_hit:
            hits += 1
        if any_inside:
            inside += 1

    hit_ratio = hits / max(1, total)
    inside_ratio = inside / max(1, total)
    return float(hit_ratio), float(min_dist), float(inside_ratio)


def connector_vote_features(
    conn_cache: ConnectorLayerCache,
    xs: np.ndarray,
    ys: np.ndarray,
    sample_step: int,
    radius_m: float,
) -> Dict[str, float]:
    """
    Returns interpretable ratios:
      conn_left_ratio, conn_right_ratio, conn_straight_ratio, conn_uturn_ratio, conn_unknown_ratio, conn_none_ratio
      conn_best_ratio (best among L/R/S/U/UNK excluding NONE)
    """
    if conn_cache is None or conn_cache.gdf is None or len(conn_cache.gdf) == 0:
        return {
            "conn_left_ratio": 0.0,
            "conn_right_ratio": 0.0,
            "conn_straight_ratio": 0.0,
            "conn_uturn_ratio": 0.0,
            "conn_unknown_ratio": 0.0,
            "conn_none_ratio": 1.0,
            "conn_best_ratio": 0.0,
        }

    gdf = conn_cache.gdf
    sindex = conn_cache.sindex
    col = conn_cache.turn_col

    r = float(radius_m)
    counts = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0, "UTURN": 0, "UNKNOWN": 0, "NONE": 0}
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

    for i in range(0, len(xs), max(1, sample_step)):
        total += 1
        p = Point(float(xs[i]), float(ys[i]))

        if sindex is not None:
            bbox = (p.x - r, p.y - r, p.x + r, p.y + r)
            cand_idx = list(sindex.intersection(bbox))
            cand = gdf.iloc[cand_idx] if cand_idx else None
        else:
            cand = gdf

        if cand is None or len(cand) == 0:
            counts["NONE"] += 1
            continue

        any_hit = False
        # Vote any connector within radius (could be multiple per point)
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
                v = row[col] if (col and col in row) else None
                counts[to_label(v)] += 1

        if not any_hit:
            counts["NONE"] += 1

    # normalize by total "point samples"
    # (note: LEFT/RIGHT/etc are "votes", could exceed total; still interpretable as density)
    denom = max(1, total)
    left = counts["LEFT"] / denom
    right = counts["RIGHT"] / denom
    straight = counts["STRAIGHT"] / denom
    uturn = counts["UTURN"] / denom
    unk = counts["UNKNOWN"] / denom
    none = counts["NONE"] / denom

    # best ratio among non-NONE buckets (normalized by their sum for interpretability)
    denom2 = max(1e-9, (left + right + straight + uturn + unk))
    best_ratio = max(left, right, straight, uturn, unk) / denom2

    return {
        "conn_left_ratio": float(left),
        "conn_right_ratio": float(right),
        "conn_straight_ratio": float(straight),
        "conn_uturn_ratio": float(uturn),
        "conn_unknown_ratio": float(unk),
        "conn_none_ratio": float(none),
        "conn_best_ratio": float(best_ratio),
    }


# -----------------------------
# Trajectory interpretable features
# -----------------------------
def trajectory_features(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Dict[str, float]:
    if len(xs) < 3:
        return {
            "travel_dist_m": 0.0,
            "net_heading_deg": 0.0,
            "abs_net_heading_deg": 0.0,
            "total_abs_heading_deg": 0.0,
            "mean_step_dist_m": 0.0,
            "stationary_ratio": 1.0,
            "x_end_ego": 0.0,
            "y_end_ego": 0.0,
        }

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    travel = float(math.hypot(dx, dy))

    dh_net = wrap_to_pi(float(hs[-1] - hs[0]))
    net_deg = float(math.degrees(dh_net))
    abs_net_deg = abs(net_deg)

    dh = np.diff(hs)
    dh = np.vectorize(wrap_to_pi)(dh)
    total_abs_deg = float(math.degrees(float(np.sum(np.abs(dh)))))

    # speed proxy from step distances (not true m/s unless dt known; still useful)
    step_dx = np.diff(xs)
    step_dy = np.diff(ys)
    step_dist = np.sqrt(step_dx ** 2 + step_dy ** 2)
    mean_step_dist = float(np.mean(step_dist))

    # stationary ratio: fraction of steps with tiny movement
    stationary_ratio = float(np.mean(step_dist < 0.25))  # 25cm per step threshold

    x_end_ego, y_end_ego = endpoint_in_ego_frame(xs, ys, hs)

    return {
        "travel_dist_m": float(travel),
        "net_heading_deg": float(net_deg),
        "abs_net_heading_deg": float(abs_net_deg),
        "total_abs_heading_deg": float(total_abs_deg),
        "mean_step_dist_m": float(mean_step_dist),
        "stationary_ratio": float(stationary_ratio),
        "x_end_ego": float(x_end_ego),
        "y_end_ego": float(y_end_ego),
    }


# -----------------------------
# Scenario loading
# -----------------------------
def build_scenarios(split: str, max_scenarios: int, num_workers: int) -> List[Any]:
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]
    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(f"Cannot find DB at {db_root}")

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

    scenarios = builder.get_scenarios(scenario_filter, worker)
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]  # hard cap for correct tqdm denominator
    return scenarios


# -----------------------------
# Main clustering pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_scenarios", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)

    parser.add_argument("--map_sample_step", type=int, default=5)
    parser.add_argument("--intersection_tol_m", type=float, default=12.0)

    parser.add_argument("--connector_sample_step", type=int, default=5)
    parser.add_argument("--connector_radius_m", type=float, default=5.0)

    parser.add_argument("--exemplars_per_cluster", type=int, default=12)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "cluster_labels.jsonl"
    tokens_path = out_dir / "tokens.txt"
    feats_path = out_dir / "features.npy"
    exemplars_path = out_dir / "exemplars.json"

    print(f"[INFO] split={args.split}")
    print(f"[INFO] max_scenarios={args.max_scenarios}")
    print(f"[INFO] num_clusters={args.num_clusters}")
    print(f"[INFO] out_dir={out_dir}")

    print("[INFO] Loading scenarios...")
    scenarios = build_scenarios(args.split, args.max_scenarios, args.num_workers)
    print(f"[INFO] Loaded {len(scenarios)} scenarios (after hard cap).")

    map_cache = MapLayerCache()

    tokens: List[str] = []
    feat_rows: List[List[float]] = []
    feat_meta: List[Dict[str, Any]] = []  # for writing jsonl later

    # Feature vector definition (keep order fixed!)
    FEAT_NAMES = [
        # motion / heading
        "travel_dist_m",
        "mean_step_dist_m",
        "stationary_ratio",
        "abs_net_heading_deg",
        "total_abs_heading_deg",
        # endpoint in ego frame
        "x_end_ego",
        "y_end_ego",
        # intersection semantics
        "inter_hit_ratio",
        "inter_inside_ratio",
        "inter_min_dist_m",
        # connector evidence (ratios)
        "conn_left_ratio",
        "conn_right_ratio",
        "conn_straight_ratio",
        "conn_uturn_ratio",
        "conn_unknown_ratio",
        "conn_none_ratio",
        "conn_best_ratio",
        # light tag/type flags (binary)
        "tag_intersection",
        "tag_right_turn",
        "tag_roundabout",
        "tag_uturn",
        "tag_pudo",
    ]

    def flags_from_tags_and_type(stype: str, tags: List[str]) -> Dict[str, float]:
        st = _upper(stype)
        t = " ".join(tags)
        tag_intersection = 1.0 if (contains_any(["INTERSECTION", "TRAFFIC_LIGHT", "STOP_SIGN"], st) or contains_any(["INTERSECTION", "TRAFFIC_LIGHT", "STOP_SIGN"], t)) else 0.0
        tag_right_turn = 1.0 if (("RIGHT" in st and "TURN" in st) or ("STARTING_RIGHT" in st and "TURN" in st) or ("RIGHT" in t and "TURN" in t)) else 0.0
        tag_roundabout = 1.0 if ("ROUNDABOUT" in st or "ROUNDABOUT" in t) else 0.0
        tag_uturn = 1.0 if ("UTURN" in st or "U_TURN" in st or "UTURN" in t or "U_TURN" in t) else 0.0
        tag_pudo = 1.0 if ("PICKUP_DROPOFF" in st or "PICKUP_DROPOFF" in t or "PUDO" in st or "PUDO" in t) else 0.0
        return {
            "tag_intersection": tag_intersection,
            "tag_right_turn": tag_right_turn,
            "tag_roundabout": tag_roundabout,
            "tag_uturn": tag_uturn,
            "tag_pudo": tag_pudo,
        }

    print("[INFO] Computing features (trajectory + SemanticMapLayer semantics)...")
    for scenario in tqdm(scenarios, total=len(scenarios), desc="Feature extraction"):
        tok = scenario.token
        stype = getattr(scenario, "scenario_type", "")
        tags = get_tags_if_available(scenario)

        xs, ys, hs = compute_ego_xyh(scenario)
        traj_f = trajectory_features(xs, ys, hs)

        # intersection layer features
        inter_cache = map_cache.get_intersection_cache(scenario)
        inter_hit_ratio, inter_min_dist_m, inter_inside_ratio = (0.0, float("inf"), 0.0)
        if inter_cache is not None:
            inter_hit_ratio, inter_min_dist_m, inter_inside_ratio = intersection_features(
                inter_cache, xs, ys, sample_step=args.map_sample_step, tol_m=args.intersection_tol_m
            )

        # connector layer vote features
        conn_cache = map_cache.get_connector_cache(scenario)
        conn_f = connector_vote_features(
            conn_cache, xs, ys, sample_step=args.connector_sample_step, radius_m=args.connector_radius_m
        )

        flags = flags_from_tags_and_type(stype, tags)

        # Assemble numeric vector (replace inf with a large sentinel for scaler stability)
        inter_min_dist_num = float(inter_min_dist_m)
        if not np.isfinite(inter_min_dist_num):
            inter_min_dist_num = 1e6

        row = [
            traj_f["travel_dist_m"],
            traj_f["mean_step_dist_m"],
            traj_f["stationary_ratio"],
            traj_f["abs_net_heading_deg"],
            traj_f["total_abs_heading_deg"],
            traj_f["x_end_ego"],
            traj_f["y_end_ego"],
            float(inter_hit_ratio),
            float(inter_inside_ratio),
            float(inter_min_dist_num),
            conn_f["conn_left_ratio"],
            conn_f["conn_right_ratio"],
            conn_f["conn_straight_ratio"],
            conn_f["conn_uturn_ratio"],
            conn_f["conn_unknown_ratio"],
            conn_f["conn_none_ratio"],
            conn_f["conn_best_ratio"],
            flags["tag_intersection"],
            flags["tag_right_turn"],
            flags["tag_roundabout"],
            flags["tag_uturn"],
            flags["tag_pudo"],
        ]

        tokens.append(str(tok))
        feat_rows.append(row)

        # keep a small meta snapshot for jsonl (still interpretable)
        feat_meta.append({
            "scenario_type": stype,
            "tags": tags,
            "features": dict(zip(FEAT_NAMES, row)),
        })

    X = np.asarray(feat_rows, dtype=np.float32)
    print(f"[INFO] Feature matrix: X.shape={X.shape} (N x D)")

    # Save raw features + tokens now (useful even before clustering)
    np.save(feats_path, X)
    tokens_path.write_text("\n".join(tokens))
    (out_dir / "feature_names.txt").write_text("\n".join(FEAT_NAMES))
    print(f"[INFO] Saved features to: {feats_path}")
    print(f"[INFO] Saved tokens to:   {tokens_path}")

    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("[INFO] Clustering with MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=int(args.num_clusters),
        random_state=int(args.kmeans_seed),
        batch_size=int(args.batch_size),
        n_init="auto" if hasattr(MiniBatchKMeans, "n_init") else 10,
        reassignment_ratio=0.01,
    )
    cluster_ids = kmeans.fit_predict(Xs)
    centers = kmeans.cluster_centers_  # in scaled space

    # ---------
    # Exemplars: closest-to-centroid tokens per cluster
    # ---------
    print("[INFO] Selecting exemplars (closest to centroid)...")
    # squared distance to assigned centroid
    d2 = np.sum((Xs - centers[cluster_ids]) ** 2, axis=1)

    exemplars: Dict[str, Any] = {}
    for c in range(int(args.num_clusters)):
        idx = np.where(cluster_ids == c)[0]
        if idx.size == 0:
            exemplars[str(c)] = {"count": 0, "tokens": []}
            continue
        # sort cluster members by distance
        order = idx[np.argsort(d2[idx])]
        topk = order[: int(args.exemplars_per_cluster)]
        exemplars[str(c)] = {
            "count": int(idx.size),
            "tokens": [tokens[i] for i in topk],
        }

    exemplars_path.write_text(json.dumps(exemplars, indent=2))
    print(f"[INFO] Saved exemplars to: {exemplars_path}")

    # ---------
    # Write cluster_labels.jsonl
    # ---------
    print("[INFO] Writing cluster_labels.jsonl ...")
    with labels_path.open("w") as f:
        for i, tok in enumerate(tokens):
            rec = {
                "token": tok,
                "cluster_id": int(cluster_ids[i]),
                "features": feat_meta[i]["features"],  # interpretable snapshot
                "scenario_type": feat_meta[i]["scenario_type"],
                "tags": feat_meta[i]["tags"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"[INFO] Saved labels to: {labels_path}")

    # ---------
    # Plots
    # ---------
    print("[INFO] Making plots...")
    # cluster counts bar
    counts = np.bincount(cluster_ids, minlength=int(args.num_clusters))
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.xlabel("cluster_id")
    plt.ylabel("count")
    plt.title("Cluster counts")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "cluster_counts.png", dpi=160)
    plt.close()

    # PCA scatter (2D)
    pca = PCA(n_components=2, random_state=int(args.kmeans_seed))
    Xp = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 6))
    plt.scatter(Xp[:, 0], Xp[:, 1], s=3, c=cluster_ids)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter colored by cluster_id")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "pca_scatter.png", dpi=160)
    plt.close()

    # Feature importance proxy per cluster: show top differing features (mean in scaled space)
    # Save a small text report (interpretable)
    report_lines = []
    report_lines.append("Cluster feature profile (means in *scaled* feature space):")
    report_lines.append(f"Feature order:\n  " + ", ".join(FEAT_NAMES))
    report_lines.append("")
    for c in range(int(args.num_clusters)):
        idx = np.where(cluster_ids == c)[0]
        if idx.size == 0:
            continue
        mu = np.mean(Xs[idx], axis=0)
        # top 6 absolute mean features
        top = np.argsort(np.abs(mu))[::-1][:6]
        report_lines.append(f"cluster {c:02d}  count={idx.size}")
        for j in top:
            report_lines.append(f"  {FEAT_NAMES[j]:24s}  mean_scaled={mu[j]:+.3f}")
        report_lines.append("")
    (out_dir / "plots" / "cluster_feature_report.txt").write_text("\n".join(report_lines))

    print(f"[DONE] Outputs in: {out_dir}")
    print(f"  - {labels_path.name}")
    print(f"  - {feats_path.name}, {tokens_path.name}, feature_names.txt")
    print(f"  - exemplars.json")
    print(f"  - plots/cluster_counts.png, plots/pca_scatter.png, plots/cluster_feature_report.txt")


if __name__ == "__main__":
    main()
