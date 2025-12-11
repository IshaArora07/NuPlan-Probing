#!/usr/bin/env python
"""
Visualize nuPlan scenarios *by EMoE class* using scene_labels.jsonl.

Run example:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

  python visualize_by_scene_labels.py \
    --split mini \
    --scene_labels /path/to/scene_labels.jsonl \
    --out_dir ./viz_by_class \
    --num_per_class 30 \
    --max_scenarios_scan 50000 \
    --map_radius 90
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# nuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


# ---------------------------------------------------------------------
# EMoE names
# ---------------------------------------------------------------------
EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
    "others",                         # 6
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_ego_xyh(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ego rear-axle x, y, heading over all iterations as numpy arrays."""
    xs_list: List[float] = []
    ys_list: List[float] = []
    hs_list: List[float] = []

    n_iter = scenario.get_number_of_iterations()
    for i in range(n_iter):
        ego = scenario.get_ego_state_at_iteration(i)
        xs_list.append(ego.rear_axle.x)
        ys_list.append(ego.rear_axle.y)
        hs_list.append(float(ego.rear_axle.heading))

    xs = np.asarray(xs_list, dtype=float)
    ys = np.asarray(ys_list, dtype=float)
    hs = np.asarray(hs_list, dtype=float)
    return xs, ys, hs


def to_ego_frame(xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform global ego positions into ego frame at t=0:
      - translate so that (x0, y0) -> (0, 0)
      - rotate so that heading0 aligns with +x axis
    """
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


# ---------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------
def build_scenarios(split: str, limit_total_scenarios: Optional[int]) -> List[Any]:
    """
    Load nuPlan scenarios for the given split.
    NOTE: limit_total_scenarios is a soft cap in some devkit versions;
          we also hard-stop scanning in main().
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

    data_root = Path(data_root).expanduser().resolve()
    map_root = Path(map_root).expanduser().resolve()

    db_root = data_root / "nuplan-v1.1" / "splits" / split
    if not db_root.exists():
        raise FileNotFoundError(
            f"Cannot find DB at {db_root}. Check NUPLAN_DATA_ROOT and split name."
        )

    worker = SingleMachineParallelExecutor(
        use_process_pool=False,
        num_workers=8,
    )

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
        max_workers=8,
    )

    scenarios = builder.get_scenarios(scenario_filter, worker)
    return scenarios


# ---------------------------------------------------------------------
# Map layers (same idea as your improved global view)
# ---------------------------------------------------------------------
def get_vector_layers(map_api):
    """
    Access several vector GeoDataFrames from map_api:
      - LANE
      - LANE_CONNECTOR
      - DRIVABLE_AREA
      - ROADBLOCK
      - INTERSECTION
      - BOUNDARIES
    """
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
    """Filter GeoDataFrame to a square window centered at (cx, cy) with side 2*radius."""
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


def plot_local_map(
    ax,
    lane_gdf_local,
    conn_gdf_local,
    driv_gdf_local,
    rb_gdf_local,
    inter_gdf_local,
    bound_gdf_local,
):
    """
    Plot local map geometry:
      - DRIVABLE_AREA polygons as light filled patches
      - ROADBLOCK outlines
      - INTERSECTION polygons outlines
      - BOUNDARIES as thin lines
      - LANE polylines
      - LANE_CONNECTOR polylines dashed
    """
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


# ---------------------------------------------------------------------
# NEW (CHANGED): scene_labels.jsonl loading + token sampling
# ---------------------------------------------------------------------
def load_scene_labels(scene_labels_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load scene_labels.jsonl into:
      token -> {"emoe_class_id": int, "emoe_class_name": str, "stage": optional}
    """
    token_to_info: Dict[str, Dict[str, Any]] = {}
    with scene_labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            token = obj.get("token", None)
            cid = obj.get("emoe_class_id", None)
            if token is None or cid is None:
                continue
            cid = int(cid)
            cname = obj.get("emoe_class_name", EMOE_SCENE_TYPES[cid] if 0 <= cid < 7 else "unknown")
            token_to_info[token] = {
                "emoe_class_id": cid,
                "emoe_class_name": cname,
                "stage": obj.get("stage", None),
            }
    return token_to_info


def choose_tokens_per_class(
    token_to_info: Dict[str, Dict[str, Any]],
    num_per_class: int,
    seed: int = 0,
) -> Dict[int, List[str]]:
    """
    Randomly choose up to num_per_class tokens for each EMoE class.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[str]] = {c: [] for c in range(7)}
    for tok, info in token_to_info.items():
        cid = int(info["emoe_class_id"])
        if 0 <= cid < 7:
            by_class[cid].append(tok)

    chosen: Dict[int, List[str]] = {}
    for c in range(7):
        toks = by_class[c]
        if not toks:
            chosen[c] = []
            continue
        if len(toks) <= num_per_class:
            chosen[c] = toks
        else:
            idx = rng.choice(len(toks), size=num_per_class, replace=False)
            chosen[c] = [toks[i] for i in idx]
    return chosen


# ---------------------------------------------------------------------
# CHANGED: visualization (ONLY world + ego; no connector plots/summary)
# ---------------------------------------------------------------------
def visualize_scenario_two_views(
    scenario,
    out_path: Path,
    emoe_class_id: int,
    emoe_class_name: str,
    stage: Optional[str],
    map_radius: float = 60.0,
):
    """
    Create a combined figure for one scenario:
      (A) Global map view
      (B) Ego-frame view
    Includes heading + distance info.
    """
    stype = scenario.scenario_type
    token = scenario.token
    map_api = scenario.map_api

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

    lane_gdf, conn_gdf, driv_gdf, rb_gdf, inter_gdf, bound_gdf = get_vector_layers(map_api)
    if lane_gdf is None and conn_gdf is None and driv_gdf is None:
        return False

    lane_gdf_local = filter_local_window(lane_gdf, cx, cy, map_radius)
    conn_gdf_local = filter_local_window(conn_gdf, cx, cy, map_radius)
    driv_gdf_local = filter_local_window(driv_gdf, cx, cy, map_radius)
    rb_gdf_local = filter_local_window(rb_gdf, cx, cy, map_radius)
    inter_gdf_local = filter_local_window(inter_gdf, cx, cy, map_radius)
    bound_gdf_local = filter_local_window(bound_gdf, cx, cy, map_radius)

    fig = plt.figure(figsize=(13, 6))

    # (A) Global map view
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
        f"class={emoe_class_id}:{emoe_class_name}\n"
        f"scenario_type={stype}"
    )
    ax1.legend(loc="best")

    # (B) Ego-frame view
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(xs_local, ys_local, "-o", markersize=2.5, linewidth=1.0)
    ax2.plot(0.0, 0.0, "go", label="start")
    ax2.plot(xs_local[-1], ys_local[-1], "rs", label="end")

    # Heading arrow (net heading change)
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

    # Text block (distance + heading + token + optional stage)
    stage_str = f"\nstage={stage}" if stage else ""
    ax2.set_title(
        "Ego-frame trajectory\n"
        f"Δheading={dh_deg:+.1f}° (|Δ|={abs_dh_deg:.1f}°), dist={dist:.1f}m\n"
        f"token={token}{stage_str}"
    )
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------
# Main (CHANGED): visualize N per class using scene_labels.jsonl
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini", help="nuPlan split: mini, trainval, test")
    parser.add_argument("--scene_labels", type=str, required=True, help="Path to scene_labels.jsonl")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (class subfolders created)")
    parser.add_argument("--num_per_class", type=int, default=20, help="How many scenarios to visualize per class")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for token sampling")
    parser.add_argument("--map_radius", type=float, default=80.0, help="Half-size of map window [m]")
    parser.add_argument("--max_scenarios_scan", type=int, default=50000,
                        help="Hard cap: how many scenarios to iterate while searching tokens")
    parser.add_argument("--limit_load", type=int, default=None,
                        help="Optional devkit filter limit_total_scenarios (not always strict)")

    args = parser.parse_args()

    scene_labels_path = Path(args.scene_labels).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading scene labels from: {scene_labels_path}")
    token_to_info = load_scene_labels(scene_labels_path)
    print(f"[INFO] Loaded labels for {len(token_to_info)} tokens.")

    print(f"[INFO] Sampling up to {args.num_per_class} tokens per class (seed={args.seed})")
    chosen_by_class = choose_tokens_per_class(token_to_info, args.num_per_class, seed=args.seed)

    # Create folders per class
    for c in range(7):
        class_dir = out_root / f"{c:01d}_{EMOE_SCENE_TYPES[c]}"
        class_dir.mkdir(parents=True, exist_ok=True)

    # Build a global target set of tokens we want to visualize
    target_tokens: Set[str] = set()
    for c in range(7):
        target_tokens.update(chosen_by_class[c])

    target_total = len(target_tokens)
    print(f"[INFO] Total target tokens to visualize: {target_total}")
    for c in range(7):
        print(f"  - class {c} {EMOE_SCENE_TYPES[c]:28s}: {len(chosen_by_class[c])} targets")

    if target_total == 0:
        print("[WARN] No target tokens found (scene_labels may be empty or paths wrong). Exiting.")
        return

    # Count how many we’ve successfully visualized per class
    done_by_class: Dict[int, int] = {c: 0 for c in range(7)}
    done_tokens: Set[str] = set()

    # Load scenarios
    print(f"[INFO] Loading scenarios from split='{args.split}' ...")
    scenarios = build_scenarios(args.split, limit_total_scenarios=args.limit_load)
    print(f"[INFO] Loaded {len(scenarios)} scenarios (devkit load). Will scan up to {args.max_scenarios_scan}.")

    scanned = 0
    saved = 0

    for scenario in tqdm(scenarios, desc="Scanning scenarios for target tokens"):
        scanned += 1
        if args.max_scenarios_scan is not None and scanned > args.max_scenarios_scan:
            break

        tok = scenario.token
        if tok not in target_tokens:
            continue
        if tok in done_tokens:
            continue

        info = token_to_info.get(tok, None)
        if info is None:
            continue

        cid = int(info["emoe_class_id"])
        cname = str(info.get("emoe_class_name", EMOE_SCENE_TYPES[cid] if 0 <= cid < 7 else "unknown"))
        stage = info.get("stage", None)

        # Save path under class folder
        class_dir = out_root / f"{cid:01d}_{EMOE_SCENE_TYPES[cid]}"
        idx = done_by_class.get(cid, 0)
        out_path = class_dir / f"{idx:03d}_token_{tok}.png"

        ok = visualize_scenario_two_views(
            scenario=scenario,
            out_path=out_path,
            emoe_class_id=cid,
            emoe_class_name=cname,
            stage=stage,
            map_radius=args.map_radius,
        )
        if not ok:
            continue

        done_tokens.add(tok)
        done_by_class[cid] = idx + 1
        saved += 1

        # Early stop: if we’ve reached quota for every class (or exhausted)
        all_met = True
        for c in range(7):
            if done_by_class[c] < len(chosen_by_class[c]):
                all_met = False
                break
        if all_met:
            break

    print("\n[INFO] Done.")
    print(f"[INFO] Scanned scenarios: {scanned}")
    print(f"[INFO] Saved figures:     {saved}")
    print("[INFO] Saved per class:")
    for c in range(7):
        want = len(chosen_by_class[c])
        got = done_by_class[c]
        print(f"  - class {c} {EMOE_SCENE_TYPES[c]:28s}: {got}/{want}")

    print(f"[INFO] Output root: {out_root}")


if __name__ == "__main__":
    main()
