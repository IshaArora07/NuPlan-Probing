#!/usr/bin/env python3
"""
Analyze KMeans clustering outputs AND visualize exemplar tokens directly.

You said:
- You don't have scene_labels.jsonl
- You ran KMeans.py and have its outputs
- You want exemplars per cluster visualized (global + ego view) AFTER KMeans

This script:
1) Loads KMeans assignments: token -> cluster_id
   Supported formats:
     A) JSONL: each line has {"token": "...", "cluster_id": int, ...}
     B) CSV: token,cluster_id columns
     C) NPY pairs: --tokens_npy and --labels_npy

2) Chooses exemplars per cluster:
   - If you provide --features_npy and --centroids_npy: chooses closest-to-centroid tokens.
   - Otherwise: randomly samples tokens per cluster.

3) Scans nuPlan scenarios and visualizes ONLY the exemplar tokens.
   - Global HD map view (drivable_area, roadblocks, intersections, boundaries, lanes, lane_connectors)
   - Ego-frame trajectory
   - Saves PNGs into cluster folders

Run:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python analyze_kmeans_and_visualize_exemplars.py \
    --split trainval \
    --out_dir ./kmeans_exemplar_viz \
    --assignments_jsonl ./kmeans_assignments.jsonl \
    --num_exemplars 30 \
    --max_scenarios_scan 200000 \
    --map_radius 90

Optional (for true exemplars):
  --features_npy ./X_features.npy --centroids_npy ./kmeans_centers.npy
  (features rows MUST align with tokens order in assignments file or tokens_npy)
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


# -----------------------------
# Helpers: geometry
# -----------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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

    return (
        np.asarray(xs_list, dtype=float),
        np.asarray(ys_list, dtype=float),
        np.asarray(hs_list, dtype=float),
    )


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


# -----------------------------
# nuPlan scenario loading
# -----------------------------
def build_scenarios(split: str, limit_total_scenarios: Optional[int], num_workers: int = 8) -> List[Any]:
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
        raise FileNotFoundError(f"Cannot find DB at {db_root}")

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

    scenarios = builder.get_scenarios(scenario_filter, worker)
    return scenarios


# -----------------------------
# Map layers for richer global view
# -----------------------------
def get_vector_layers(map_api):
    lane_gdf = conn_gdf = driv_gdf = rb_gdf = inter_gdf = bound_gdf = None

    def _try(layer):
        try:
            return map_api._get_vector_map_layer(layer)  # type: ignore[attr-defined]
        except Exception:
            return None

    lane_gdf = _try(SemanticMapLayer.LANE)
    conn_gdf = _try(SemanticMapLayer.LANE_CONNECTOR)
    driv_gdf = _try(SemanticMapLayer.DRIVABLE_AREA)
    rb_gdf = _try(SemanticMapLayer.ROADBLOCK)
    inter_gdf = _try(SemanticMapLayer.INTERSECTION)
    bound_gdf = _try(SemanticMapLayer.BOUNDARIES)

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


def plot_local_map(ax, lane_gdf_local, conn_gdf_local, driv_gdf_local, rb_gdf_local, inter_gdf_local, bound_gdf_local):
    # Drivable area fill
    if driv_gdf_local is not None:
        for geom in driv_gdf_local.geometry:
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

    # Roadblocks
    if rb_gdf_local is not None:
        for geom in rb_gdf_local.geometry:
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

    # Intersections
    if inter_gdf_local is not None:
        for geom in inter_gdf_local.geometry:
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
    if bound_gdf_local is not None:
        for geom in bound_gdf_local.geometry:
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
    if lane_gdf_local is not None:
        for geom in lane_gdf_local.geometry:
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
    if conn_gdf_local is not None:
        for geom in conn_gdf_local.geometry:
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


# -----------------------------
# Visualization (global + ego)
# -----------------------------
def visualize_scenario_two_views(
    scenario,
    out_path: Path,
    cluster_id: int,
    exemplar_rank: int,
    map_radius: float,
):
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
    if lane_gdf is None and conn_gdf is None and driv_gdf is None:
        return False

    lane_gdf_local = filter_local_window(lane_gdf, cx, cy, map_radius)
    conn_gdf_local = filter_local_window(conn_gdf, cx, cy, map_radius)
    driv_gdf_local = filter_local_window(driv_gdf, cx, cy, map_radius)
    rb_gdf_local = filter_local_window(rb_gdf, cx, cy, map_radius)
    inter_gdf_local = filter_local_window(inter_gdf, cx, cy, map_radius)
    bound_gdf_local = filter_local_window(bound_gdf, cx, cy, map_radius)

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    plot_local_map(ax1, lane_gdf_local, conn_gdf_local, driv_gdf_local, rb_gdf_local, inter_gdf_local, bound_gdf_local)
    ax1.plot(xs, ys, "-", linewidth=1.2)
    ax1.plot(xs[0], ys[0], "go", markersize=6, label="start")
    ax1.plot(xs[-1], ys[-1], "rs", markersize=6, label="end")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(cx - map_radius, cx + map_radius)
    ax1.set_ylim(cy - map_radius, cy + map_radius)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(
        "Global map view\n"
        f"cluster={cluster_id} exemplar_rank={exemplar_rank}\n"
        f"scenario_type={getattr(scenario, 'scenario_type', '')}"
    )
    ax1.legend(loc="best")

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
        "Ego-frame trajectory\n"
        f"Δheading={dh_deg:+.1f}° (|Δ|={abs_dh_deg:.1f}°), dist={dist:.1f}m\n"
        f"token={scenario.token}"
    )
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# -----------------------------
# Load KMeans assignments
# -----------------------------
def load_assignments_jsonl(path: Path) -> Tuple[List[str], np.ndarray]:
    tokens: List[str] = []
    labels: List[int] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tok = obj.get("token")
            cid = obj.get("cluster_id", obj.get("cluster", obj.get("label")))
            if tok is None or cid is None:
                continue
            tokens.append(str(tok))
            labels.append(int(cid))
    return tokens, np.asarray(labels, dtype=int)


def load_assignments_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    import csv
    tokens: List[str] = []
    labels: List[int] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "token" not in reader.fieldnames or ("cluster_id" not in reader.fieldnames and "label" not in reader.fieldnames):
            raise ValueError(f"CSV must have columns token and cluster_id (or label). Found: {reader.fieldnames}")
        for row in reader:
            tok = row["token"]
            cid = row.get("cluster_id", row.get("label"))
            if tok is None or cid is None:
                continue
            tokens.append(str(tok))
            labels.append(int(cid))
    return tokens, np.asarray(labels, dtype=int)


def choose_exemplars(
    tokens: List[str],
    labels: np.ndarray,
    num_exemplars: int,
    seed: int,
    features: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Returns: cluster_id -> list[(token, score)] where score is distance-to-centroid (lower = better)
    If features+centroids provided: closest-to-centroid exemplars
    Else: random exemplars (score = NaN)
    """
    rng = np.random.default_rng(seed)
    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        by_cluster[int(c)].append(i)

    exemplars: Dict[int, List[Tuple[str, float]]] = {}

    use_dist = (features is not None) and (centroids is not None)
    if use_dist:
        if features.shape[0] != len(tokens):
            raise ValueError(f"features rows ({features.shape[0]}) must match tokens ({len(tokens)}).")
        if centroids.ndim != 2:
            raise ValueError("centroids must be [K, D].")

    for c, idxs in by_cluster.items():
        if not idxs:
            exemplars[c] = []
            continue

        if use_dist:
            mu = centroids[c] if c < centroids.shape[0] else None
            if mu is None:
                # fallback random if centroid missing
                pick = rng.choice(idxs, size=min(num_exemplars, len(idxs)), replace=False)
                exemplars[c] = [(tokens[i], float("nan")) for i in pick.tolist()]
                continue

            Xc = features[idxs]  # [Nc, D]
            d = np.linalg.norm(Xc - mu[None, :], axis=1)  # [Nc]
            order = np.argsort(d)
            chosen = order[: min(num_exemplars, len(order))]
            exemplars[c] = [(tokens[idxs[j]], float(d[j])) for j in chosen.tolist()]
        else:
            pick = rng.choice(idxs, size=min(num_exemplars, len(idxs)), replace=False)
            exemplars[c] = [(tokens[i], float("nan")) for i in pick.tolist()]

    return exemplars


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="trainval", help="nuPlan split folder under nuplan-v1.1/splits/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for analysis + visualizations")

    # one of these must be provided:
    parser.add_argument("--assignments_jsonl", type=str, default="", help="JSONL with {token, cluster_id}")
    parser.add_argument("--assignments_csv", type=str, default="", help="CSV with token,cluster_id")
    parser.add_argument("--tokens_npy", type=str, default="", help="tokens.npy (dtype=str or object)")
    parser.add_argument("--labels_npy", type=str, default="", help="labels.npy (int)")

    # optional for TRUE exemplars
    parser.add_argument("--features_npy", type=str, default="", help="features.npy aligned with tokens")
    parser.add_argument("--centroids_npy", type=str, default="", help="centroids.npy [K,D]")

    parser.add_argument("--num_exemplars", type=int, default=30, help="How many exemplars to visualize per cluster")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map_radius", type=float, default=90.0)
    parser.add_argument("--max_scenarios_scan", type=int, default=200000, help="Hard cap of scenario iteration")
    parser.add_argument("--limit_load", type=int, default=None, help="Optional devkit load cap (not always strict)")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- load assignments ---
    tokens: List[str]
    labels: np.ndarray

    if args.assignments_jsonl:
        tokens, labels = load_assignments_jsonl(Path(args.assignments_jsonl))
    elif args.assignments_csv:
        tokens, labels = load_assignments_csv(Path(args.assignments_csv))
    elif args.tokens_npy and args.labels_npy:
        tokens_arr = np.load(args.tokens_npy, allow_pickle=True)
        labels = np.load(args.labels_npy)
        tokens = [str(x) for x in tokens_arr.tolist()]
        labels = np.asarray(labels, dtype=int)
    else:
        raise RuntimeError("Provide either --assignments_jsonl, --assignments_csv, or (--tokens_npy and --labels_npy).")

    if len(tokens) != labels.shape[0]:
        raise ValueError(f"tokens length {len(tokens)} != labels length {labels.shape[0]}")

    K = int(labels.max() + 1) if labels.size > 0 else 0
    counts = Counter(labels.tolist())
    print(f"[INFO] Loaded assignments: N={len(tokens)}  K~={K}")
    print("[INFO] Top cluster sizes:", counts.most_common(10))

    # optional features/centroids
    features = None
    centroids = None
    if args.features_npy and args.centroids_npy:
        features = np.load(args.features_npy)
        centroids = np.load(args.centroids_npy)
        print(f"[INFO] Loaded features:  {features.shape}")
        print(f"[INFO] Loaded centroids: {centroids.shape}")

    # --- choose exemplars ---
    exemplars = choose_exemplars(
        tokens=tokens,
        labels=labels,
        num_exemplars=args.num_exemplars,
        seed=args.seed,
        features=features,
        centroids=centroids,
    )

    # Save exemplar list for reproducibility
    exemplars_path = out_root / "exemplars.json"
    exemplars_serializable = {
        str(c): [{"token": t, "dist_to_centroid": d} for (t, d) in lst]
        for c, lst in exemplars.items()
    }
    exemplars_path.write_text(json.dumps(exemplars_serializable, indent=2))
    print(f"[INFO] Saved exemplar list to: {exemplars_path}")

    # Build target token set
    target_tokens: Set[str] = set()
    token_to_cluster_rank: Dict[str, Tuple[int, int]] = {}
    for c, lst in exemplars.items():
        cluster_dir = out_root / f"cluster_{int(c):03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for r, (tok, _d) in enumerate(lst):
            target_tokens.add(tok)
            token_to_cluster_rank[tok] = (int(c), int(r))

    print(f"[INFO] Total exemplar tokens to visualize: {len(target_tokens)}")
    if len(target_tokens) == 0:
        print("[WARN] No exemplars selected. Exiting.")
        return

    # --- load scenarios and visualize exemplars ---
    print(f"[INFO] Loading scenarios for split='{args.split}' ...")
    scenarios = build_scenarios(args.split, limit_total_scenarios=args.limit_load, num_workers=args.num_workers)
    print(f"[INFO] Devkit returned {len(scenarios)} scenarios. Will scan up to {args.max_scenarios_scan}.")

    scanned = 0
    saved = 0
    done_tokens: Set[str] = set()

    for scenario in tqdm(scenarios, desc="Scanning scenarios for exemplar tokens"):
        scanned += 1
        if args.max_scenarios_scan and scanned > args.max_scenarios_scan:
            break

        tok = scenario.token
        if tok not in target_tokens:
            continue
        if tok in done_tokens:
            continue

        c, r = token_to_cluster_rank[tok]
        out_path = out_root / f"cluster_{c:03d}" / f"{r:03d}_token_{tok}.png"

        ok = visualize_scenario_two_views(
            scenario=scenario,
            out_path=out_path,
            cluster_id=c,
            exemplar_rank=r,
            map_radius=args.map_radius,
        )
        if not ok:
            continue

        done_tokens.add(tok)
        saved += 1

        # early stop once all exemplars visualized
        if len(done_tokens) >= len(target_tokens):
            break

    print("\n[INFO] Done.")
    print(f"[INFO] Scanned scenarios: {scanned}")
    print(f"[INFO] Saved images:     {saved}/{len(target_tokens)}")
    print(f"[INFO] Output dir:       {out_root}")


if __name__ == "__main__":
    main()
