#!/usr/bin/env python3
"""
Unsupervised KMeans analysis + DIRECT visualization of exemplar tokens (in one script).

What this script does:
  1) Loads scene_labels.jsonl (no re-classification)
  2) Builds a feature matrix X from the saved "debug" fields (distance, headings, intersection, connector ratios, etc.)
  3) Runs StandardScaler + KMeans
  4) Selects exemplar tokens per cluster (closest-to-centroid)
  5) Scans nuPlan scenarios ONLY until all exemplar tokens are found (hard cap)
  6) Saves (A) Global map view + (B) Ego-frame view PNGs for each exemplar

Why this matches your request:
  ✅ Visualization is built-in to the analysis script
  ✅ Exemplar tokens are generated from clustering and visualized immediately
  ✅ Uses your richer global map drawing (SemanticMapLayer vector layers)
  ✅ Stops early once exemplars are all saved (even if max_scenarios_scan is huge)

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python unsupervised_kmeans_with_viz.py \
    --split mini \
    --scene_labels /path/to/scene_labels.jsonl \
    --out_dir ./unsup_kmeans_viz \
    --n_clusters 20 \
    --exemplars_per_cluster 10 \
    --max_scenarios_scan 500000 \
    --map_radius 90
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------------------
# Load scene_labels.jsonl and build features
# --------------------------------------------------------------------------------------
def load_scene_labels(scene_labels_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with scene_labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_feature_matrix(rows: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Build a feature vector per token from fields in scene_labels.jsonl.

    We keep features:
      - travel_distance_m
      - abs_delta_heading_deg
      - total_abs_heading_deg
      - has_intersection_tag (0/1)
      - has_intersection_map (0/1)
      - intersection_min_dist_m (clipped)
      - connector ratios: left/right/straight/uturn/unknown/none
      - connector_best_ratio
      - connector_best_type (one-hot over {LEFT,RIGHT,STRAIGHT,UTURN,UNKNOWN,NONE})

    Note:
      - If a field is missing, we fill with 0 or safe default.
      - This is KMeans-friendly (continuous + normalized later).
    """
    tokens: List[str] = []
    feats: List[List[float]] = []

    # feature names (for debugging/printing)
    feature_names: List[str] = [
        "travel_distance_m",
        "abs_delta_heading_deg",
        "total_abs_heading_deg",
        "has_intersection_tag",
        "has_intersection_map",
        "intersection_min_dist_m_clipped",
        "conn_left_ratio",
        "conn_right_ratio",
        "conn_straight_ratio",
        "conn_uturn_ratio",
        "conn_unknown_ratio",
        "conn_none_ratio",
        "conn_best_ratio",
        "best_is_left",
        "best_is_right",
        "best_is_straight",
        "best_is_uturn",
        "best_is_unknown",
        "best_is_none",
    ]

    for r in rows:
        tok = r.get("token", None)
        if not tok:
            continue

        dbg = r.get("debug", {}) or {}

        travel = _safe_float(r.get("travel_distance_m", dbg.get("dist", 0.0)), 0.0)
        abs_dh = _safe_float(dbg.get("abs_delta_heading_deg", None), 0.0)
        tot_abs = _safe_float(dbg.get("total_abs_heading_deg", None), 0.0)

        has_itag = 1.0 if bool(dbg.get("has_intersection_tag", False)) else 0.0
        has_imap = 1.0 if bool(dbg.get("has_intersection_map", False)) else 0.0

        inter_min_dist = _safe_float(dbg.get("intersection_min_dist_m", 0.0), 0.0)
        # clip to keep scale sane (intersection distances can be huge if missing)
        inter_min_dist = float(np.clip(inter_min_dist, 0.0, 200.0))

        # connector counts -> ratios
        conn_counts = dbg.get("connector_counts", None)
        conn_best_type = (dbg.get("connector_best_type", None) or "NONE")
        conn_best_ratio = _safe_float(dbg.get("connector_best_ratio", 0.0), 0.0)

        # Normalize counts if available
        # expected keys: LEFT, RIGHT, STRAIGHT, UTURN, UNKNOWN, NONE
        keys = ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN", "NONE"]
        counts = {k: 0.0 for k in keys}
        if isinstance(conn_counts, dict):
            for k in keys:
                counts[k] = _safe_float(conn_counts.get(k, 0.0), 0.0)

        denom = sum(counts[k] for k in ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "UNKNOWN", "NONE"])
        denom = max(1.0, denom)
        ratios = {k: counts[k] / denom for k in keys}

        best = str(conn_best_type).upper().strip()
        best_is = {
            "best_is_left": 1.0 if best == "LEFT" else 0.0,
            "best_is_right": 1.0 if best == "RIGHT" else 0.0,
            "best_is_straight": 1.0 if best == "STRAIGHT" else 0.0,
            "best_is_uturn": 1.0 if best == "UTURN" else 0.0,
            "best_is_unknown": 1.0 if best == "UNKNOWN" else 0.0,
            "best_is_none": 1.0 if best == "NONE" else 0.0,
        }

        x = [
            travel,
            abs_dh,
            tot_abs,
            has_itag,
            has_imap,
            inter_min_dist,
            ratios["LEFT"],
            ratios["RIGHT"],
            ratios["STRAIGHT"],
            ratios["UTURN"],
            ratios["UNKNOWN"],
            ratios["NONE"],
            conn_best_ratio,
            best_is["best_is_left"],
            best_is["best_is_right"],
            best_is["best_is_straight"],
            best_is["best_is_uturn"],
            best_is["best_is_unknown"],
            best_is["best_is_none"],
        ]

        tokens.append(tok)
        feats.append(x)

    X = np.asarray(feats, dtype=np.float32)
    return tokens, X, feature_names


# --------------------------------------------------------------------------------------
# KMeans + exemplar selection
# --------------------------------------------------------------------------------------
def run_kmeans(tokens: List[str], X: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      labels: [N]
      centers: [K, D] in scaled space
      X_scaled: [N, D]
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_.astype(np.float32)

    return labels.astype(np.int32), centers, X_scaled


def choose_exemplars(
    tokens: List[str],
    labels: np.ndarray,
    centers: np.ndarray,
    X_scaled: np.ndarray,
    exemplars_per_cluster: int,
) -> Dict[int, List[str]]:
    """
    Pick exemplar tokens per cluster: closest to centroid.
    """
    K = centers.shape[0]
    exemplars: Dict[int, List[str]] = {k: [] for k in range(K)}

    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        diffs = X_scaled[idx] - centers[k]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        order = np.argsort(dists)
        chosen = idx[order[: min(exemplars_per_cluster, idx.size)]]
        exemplars[k] = [tokens[i] for i in chosen]

    return exemplars


# --------------------------------------------------------------------------------------
# nuPlan scenario loading
# --------------------------------------------------------------------------------------
def build_scenarios(split: str, limit_total_scenarios: Optional[int], num_workers: int) -> List[Any]:
    data_root = os.environ.get("NUPLAN_DATA_ROOT", None)
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", None)
    if data_root is None or map_root is None:
        raise RuntimeError(
            "Please set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT.\n"
            "Example:\n"
            "  export NUPLAN_DATA_ROOT=/path/to/nuplan\n"
            "  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps"
        )

    data_root = Path(data_root).expanduser().resolve()
    map_root = Path(map_root).expanduser().resolve()

    db_root = data_root / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(f"Cannot find DB at {db_root}. Check NUPLAN_DATA_ROOT and split.")

    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=num_workers)

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=limit_total_scenarios,
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
# Map drawing (same “rich” global view you like)
# --------------------------------------------------------------------------------------
def get_vector_layers(map_api):
    lane_gdf = conn_gdf = driv_gdf = rb_gdf = inter_gdf = bound_gdf = None

    try:
        lane_gdf = map_api._get_vector_map_layer(SemanticMapLayer.LANE)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        conn_gdf = map_api._get_vector_map_layer(SemanticMapLayer.LANE_CONNECTOR)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        driv_gdf = map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        rb_gdf = map_api._get_vector_map_layer(SemanticMapLayer.ROADBLOCK)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        inter_gdf = map_api._get_vector_map_layer(SemanticMapLayer.INTERSECTION)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        bound_gdf = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)  # type: ignore[attr-defined]
    except Exception:
        pass

    return lane_gdf, conn_gdf, driv_gdf, rb_gdf, inter_gdf, bound_gdf


def filter_local_window(gdf, cx: float, cy: float, radius: float):
    if gdf is None:
        return None
    minx = cx - radius
    maxx = cx + radius
    miny = cy - radius
    maxy = cy + radius
    try:
        return gdf.cx[minx:maxx, miny:maxy]
    except Exception:
        return gdf


def plot_local_map(ax, lane_gdf, conn_gdf, driv_gdf, rb_gdf, inter_gdf, bound_gdf):
    # Drivable area fill
    if driv_gdf is not None:
        for geom in driv_gdf.geometry:
            if geom is None:
                continue
            try:
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, alpha=0.2)
                else:
                    for g in geom.geoms:
                        xs, ys = g.exterior.xy
                        ax.fill(xs, ys, alpha=0.2)
            except Exception:
                continue

    # Roadblocks outlines
    if rb_gdf is not None:
        for geom in rb_gdf.geometry:
            if geom is None:
                continue
            try:
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    ax.plot(xs, ys, linewidth=1.0)
                else:
                    for g in geom.geoms:
                        xs, ys = g.exterior.xy
                        ax.plot(xs, ys, linewidth=1.0)
            except Exception:
                continue

    # Intersections outlines
    if inter_gdf is not None:
        for geom in inter_gdf.geometry:
            if geom is None:
                continue
            try:
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    ax.plot(xs, ys, linewidth=1.0)
                else:
                    for g in geom.geoms:
                        xs, ys = g.exterior.xy
                        ax.plot(xs, ys, linewidth=1.0)
            except Exception:
                continue

    # Boundaries
    if bound_gdf is not None:
        for geom in bound_gdf.geometry:
            if geom is None:
                continue
            try:
                xs, ys = geom.xy
                ax.plot(xs, ys, linewidth=0.3)
            except Exception:
                try:
                    for g in geom.geoms:
                        xs, ys = g.xy
                        ax.plot(xs, ys, linewidth=0.3)
                except Exception:
                    continue

    # Lanes
    if lane_gdf is not None:
        for geom in lane_gdf.geometry:
            if geom is None:
                continue
            try:
                xs, ys = geom.xy
                ax.plot(xs, ys, linewidth=0.5)
            except Exception:
                try:
                    for g in geom.geoms:
                        xs, ys = g.xy
                        ax.plot(xs, ys, linewidth=0.5)
                except Exception:
                    continue

    # Lane connectors dashed
    if conn_gdf is not None:
        for geom in conn_gdf.geometry:
            if geom is None:
                continue
            try:
                xs, ys = geom.xy
                ax.plot(xs, ys, linestyle="--", linewidth=1.0)
            except Exception:
                try:
                    for g in geom.geoms:
                        xs, ys = g.xy
                        ax.plot(xs, ys, linestyle="--", linewidth=1.0)
                except Exception:
                    continue


# --------------------------------------------------------------------------------------
# Trajectory + visualization
# --------------------------------------------------------------------------------------
def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_list: List[float] = []
    ys_list: List[float] = []
    hs_list: List[float] = []
    n_iter = scenario.get_number_of_iterations()
    for i in range(n_iter):
        ego = scenario.get_ego_state_at_iteration(i)
        xs_list.append(ego.rear_axle.x)
        ys_list.append(ego.rear_axle.y)
        hs_list.append(float(ego.rear_axle.heading))
    return np.asarray(xs_list, dtype=float), np.asarray(ys_list, dtype=float), np.asarray(hs_list, dtype=float)


def to_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(xs) == 0:
        return xs, ys
    x0, y0 = float(xs[0]), float(ys[0])
    theta0 = float(hs[0])
    x_rel = xs - x0
    y_rel = ys - y0
    c = math.cos(-theta0)
    s = math.sin(-theta0)
    x_local = c * x_rel - s * y_rel
    y_local = s * x_rel + c * y_rel
    return x_local, y_local


def visualize_cluster_exemplar(
    scenario,
    out_path: Path,
    cluster_id: int,
    *,
    map_radius: float,
    meta_text: str,
) -> bool:
    xs, ys, hs = compute_ego_xyh(scenario)
    if len(xs) < 2:
        return False

    xs_local, ys_local = to_ego_frame(xs, ys, hs)

    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    dist = float(math.hypot(dx, dy))

    dh = wrap_to_pi(float(hs[-1] - hs[0]))
    dh_deg = float(math.degrees(dh))
    abs_dh_deg = float(abs(dh_deg))

    cx = float(xs.mean())
    cy = float(ys.mean())

    lane_gdf, conn_gdf, driv_gdf, rb_gdf, inter_gdf, bound_gdf = get_vector_layers(scenario.map_api)
    lane_gdf = filter_local_window(lane_gdf, cx, cy, map_radius)
    conn_gdf = filter_local_window(conn_gdf, cx, cy, map_radius)
    driv_gdf = filter_local_window(driv_gdf, cx, cy, map_radius)
    rb_gdf = filter_local_window(rb_gdf, cx, cy, map_radius)
    inter_gdf = filter_local_window(inter_gdf, cx, cy, map_radius)
    bound_gdf = filter_local_window(bound_gdf, cx, cy, map_radius)

    fig = plt.figure(figsize=(13, 6))

    # Global
    ax1 = fig.add_subplot(1, 2, 1)
    plot_local_map(ax1, lane_gdf, conn_gdf, driv_gdf, rb_gdf, inter_gdf, bound_gdf)
    ax1.plot(xs, ys, "-", linewidth=1.2)
    ax1.plot(xs[0], ys[0], "go", markersize=6, label="start")
    ax1.plot(xs[-1], ys[-1], "rs", markersize=6, label="end")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(cx - map_radius, cx + map_radius)
    ax1.set_ylim(cy - map_radius, cy + map_radius)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(
        f"Cluster {cluster_id}  |  scenario_type={getattr(scenario, 'scenario_type', '')}\n"
        f"token={scenario.token}"
    )
    ax1.legend(loc="best")

    # Ego
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(xs_local, ys_local, "-o", markersize=2.5, linewidth=1.0)
    ax2.plot(0.0, 0.0, "go", label="start")
    ax2.plot(xs_local[-1], ys_local[-1], "rs", label="end")

    arrow_len = max(5.0, dist * 0.3)
    ax2.arrow(
        0.0, 0.0,
        arrow_len * math.cos(dh),
        arrow_len * math.sin(dh),
        head_width=1.0,
        head_length=1.0,
        linewidth=1.0,
        length_includes_head=True
    )

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x_ego [m]")
    ax2.set_ylabel("y_ego [m]")
    ax2.set_title(
        f"Ego-frame\nΔheading={dh_deg:+.1f}° (|Δ|={abs_dh_deg:.1f}°), dist={dist:.1f}m\n{meta_text}"
    )
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--scene_labels", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exemplars_per_cluster", type=int, default=10)

    parser.add_argument("--map_radius", type=float, default=90.0)
    parser.add_argument("--max_scenarios_scan", type=int, default=50000)
    parser.add_argument("--limit_load", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    scene_labels_path = Path(args.scene_labels).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading labels from: {scene_labels_path}")
    rows = load_scene_labels(scene_labels_path)
    print(f"[INFO] Loaded {len(rows)} rows.")

    tokens, X, feature_names = build_feature_matrix(rows)
    if len(tokens) == 0:
        print("[ERROR] No valid tokens found in scene_labels.jsonl. Exiting.")
        return

    print(f"[INFO] Built feature matrix: X.shape={X.shape}")
    print(f"[INFO] Features: {', '.join(feature_names)}")

    print(f"[INFO] Running KMeans: K={args.n_clusters}, seed={args.seed}")
    labels, centers, X_scaled = run_kmeans(tokens, X, n_clusters=args.n_clusters, seed=args.seed)

    # Save clustering assignments
    clusters_jsonl = out_root / "clusters.jsonl"
    with clusters_jsonl.open("w") as f:
        for tok, cid in zip(tokens, labels.tolist()):
            f.write(json.dumps({"token": tok, "cluster_id": int(cid)}) + "\n")
    print(f"[INFO] Saved cluster assignments: {clusters_jsonl}")

    exemplars = choose_exemplars(tokens, labels, centers, X_scaled, exemplars_per_cluster=args.exemplars_per_cluster)

    # Print quick stats
    counts = np.bincount(labels, minlength=args.n_clusters)
    print("\n[INFO] Cluster sizes:")
    for k in range(args.n_clusters):
        print(f"  cluster {k:02d}: {int(counts[k])}")

    # Create folders + collect target tokens
    target_tokens: Set[str] = set()
    for k in range(args.n_clusters):
        (out_root / f"cluster_{k:02d}").mkdir(parents=True, exist_ok=True)
        target_tokens.update(exemplars.get(k, []))

    print(f"\n[INFO] Total exemplar tokens to visualize: {len(target_tokens)}")
    for k in range(args.n_clusters):
        print(f"  cluster {k:02d}: {len(exemplars.get(k, []))} exemplars")

    if len(target_tokens) == 0:
        print("[WARN] No exemplars selected. Exiting.")
        return

    # Build quick token->row meta for text overlay
    token_to_meta: Dict[str, str] = {}
    for r in rows:
        tok = r.get("token", "")
        if not tok:
            continue
        dbg = r.get("debug", {}) or {}
        meta = []
        meta.append(f"stage={r.get('stage','')}")
        meta.append(f"travel={_safe_float(r.get('travel_distance_m', 0.0),0.0):.1f}m")
        meta.append(f"|Δh|={_safe_float(dbg.get('abs_delta_heading_deg',0.0),0.0):.1f}°")
        meta.append(f"tot|Δh|={_safe_float(dbg.get('total_abs_heading_deg',0.0),0.0):.1f}°")
        meta.append(f"itag={int(bool(dbg.get('has_intersection_tag',False)))} imap={int(bool(dbg.get('has_intersection_map',False)))}")
        meta.append(f"conn_best={dbg.get('connector_best_type','')}")
        token_to_meta[tok] = " | ".join(meta)

    # Scan scenarios and visualize exemplars
    print(f"\n[INFO] Loading scenarios from split='{args.split}' ...")
    scenarios = build_scenarios(args.split, limit_total_scenarios=args.limit_load, num_workers=args.num_workers)
    print(f"[INFO] Loaded {len(scenarios)} scenarios (devkit load). Will scan up to {args.max_scenarios_scan}.")

    done: Set[str] = set()
    scanned = 0
    saved = 0

    # For quick lookup: token -> cluster_id (from labels array)
    token_to_cluster: Dict[str, int] = {tok: int(cid) for tok, cid in zip(tokens, labels.tolist())}

    for sc in tqdm(scenarios, desc="Scanning scenarios for exemplars"):
        scanned += 1
        if args.max_scenarios_scan is not None and scanned > args.max_scenarios_scan:
            break

        tok = sc.token
        if tok not in target_tokens:
            continue
        if tok in done:
            continue

        cid = token_to_cluster.get(tok, None)
        if cid is None:
            continue

        class_dir = out_root / f"cluster_{cid:02d}"
        idx = sum(1 for t in done if token_to_cluster.get(t, -1) == cid)
        out_path = class_dir / f"{idx:03d}_token_{tok}.png"

        meta_text = token_to_meta.get(tok, "")
        ok = visualize_cluster_exemplar(
            sc,
            out_path=out_path,
            cluster_id=cid,
            map_radius=args.map_radius,
            meta_text=meta_text,
        )
        if not ok:
            continue

        done.add(tok)
        saved += 1

        # Early stop when all exemplars are done
        if len(done) >= len(target_tokens):
            break

    print("\n[INFO] Done.")
    print(f"[INFO] Scanned scenarios: {scanned}")
    print(f"[INFO] Saved exemplar PNGs: {saved}/{len(target_tokens)}")
    print(f"[INFO] Output dir: {out_root}")

    # Save exemplar list for reproducibility
    exemplar_json = out_root / "exemplars_by_cluster.json"
    with exemplar_json.open("w") as f:
        json.dump(exemplars, f, indent=2)
    print(f"[INFO] Saved exemplar token list: {exemplar_json}")


if __name__ == "__main__":
    main()
