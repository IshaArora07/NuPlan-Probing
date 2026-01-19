#!/usr/bin/env python3
"""
nuplan_reference_line_stats.py

Extract a lane-graph-based "reference line" for each nuPlan scenario, and compute statistics:
- start / end point (x, y)
- number of samples
- polyline length [m]
- curvature stats (mean / max) (optional but useful)

It reads:
  - NUPLAN_DATA_ROOT
  - NUPLAN_MAPS_ROOT
from the environment.

Typical dataset layout (per devkit docs):
  $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/{mini,trainval}/... .db
  $NUPLAN_DATA_ROOT/nuplan-v1.1/sensor_blobs/...
  $NUPLAN_MAPS_ROOT/{map_name}/{version}/map.gpkg

Usage examples are at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# nuPlan imports
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.utils.parallelization.single_machine_parallel_executor import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter import ScenarioFilter


# ----------------------------
# Utility + geometry
# ----------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _resolve_split_dir(data_root: Path, split: str, dataset: str = "nuplan-v1.1") -> Path:
    """
    Accepts either:
      - NUPLAN_DATA_ROOT pointing to the dataset root (e.g. ~/nuplan/dataset)
      - OR pointing directly to a split folder (e.g. .../nuplan-v1.1/splits/trainval)
    and returns the directory containing .db files.
    """
    # If user already points to a directory with .db files (or contains .db files), accept it.
    if data_root.is_dir() and any(data_root.glob("*.db")):
        return data_root

    # If they point to ".../nuplan-v1.1/splits", append split
    if data_root.name == "splits":
        candidate = data_root / split
        if candidate.is_dir():
            return candidate

    # If they point to dataset root, use docs layout: <root>/<dataset>/splits/<split>
    candidate = data_root / dataset / "splits" / split
    if candidate.is_dir():
        return candidate

    # Fallback: search a bit (robust to custom layouts)
    hits = list(data_root.rglob(f"splits/{split}"))
    for h in hits:
        if h.is_dir() and any(h.glob("*.db")):
            return h

    raise RuntimeError(
        f"Could not locate split dir for split='{split}'. "
        f"Tried:\n"
        f"  - {data_root}\n"
        f"  - {data_root / dataset / 'splits' / split}\n"
        f"  - {data_root / 'splits' / split}\n"
        f"Set NUPLAN_DATA_ROOT to the dataset root or directly to the split directory containing .db files."
    )


def _resolve_sensor_root(data_root: Path, dataset: str = "nuplan-v1.1") -> Path:
    """
    Sensor root is only needed if include_cameras=True. We still compute it robustly.
    """
    # If data_root is already inside splits/*, sensor root is typically sibling under dataset root.
    # Try common locations.
    candidates = [
        data_root / dataset / "sensor_blobs",
        data_root.parent.parent / "sensor_blobs" if data_root.name in ("mini", "trainval") and data_root.parent.name == "splits" else None,
        data_root.parent / "sensor_blobs" if data_root.name == "splits" else None,
    ]
    for c in candidates:
        if c and c.is_dir():
            return c
    # Not fatal if you don't need cameras; keep a sensible default.
    return data_root / dataset / "sensor_blobs"


def polyline_length(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    diffs = points_xy[1:] - points_xy[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def polyline_curvature(points_xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Discrete curvature estimate for a 2D polyline.
    Returns curvature per interior segment (length N-2).
    """
    if len(points_xy) < 3:
        return np.zeros((0,), dtype=np.float64)
    d1 = np.diff(points_xy, axis=0)          # (N-1, 2)
    d2 = np.diff(d1, axis=0)                 # (N-2, 2)
    num = np.abs(d1[:-1, 0] * d2[:, 1] - d1[:-1, 1] * d2[:, 0])
    den = (np.linalg.norm(d1[:-1], axis=1) ** 3) + eps
    return num / den


def _min_dist_point_to_polyline(point_xy: np.ndarray, poly_xy: np.ndarray) -> float:
    """
    Distance from point to polyline segments.
    """
    if len(poly_xy) < 2:
        return float(np.linalg.norm(point_xy - poly_xy[0])) if len(poly_xy) == 1 else float("inf")

    p = point_xy.astype(np.float64)
    a = poly_xy[:-1].astype(np.float64)
    b = poly_xy[1:].astype(np.float64)
    ab = b - a
    ap = p - a
    ab2 = np.sum(ab * ab, axis=1) + 1e-12
    t = np.clip(np.sum(ap * ab, axis=1) / ab2, 0.0, 1.0)
    proj = a + (ab.T * t).T
    d = np.linalg.norm(proj - p, axis=1)
    return float(d.min())


# ----------------------------
# Reference line extraction
# ----------------------------

@dataclass
class ReferenceLineResult:
    token: str
    log_name: str
    map_name: str
    scenario_type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    num_samples: int
    length_m: float
    curvature_mean: float
    curvature_max: float


def _lane_centerline_xy(lane_obj) -> Optional[np.ndarray]:
    """
    Try to obtain a discretized baseline/centerline for a lane object.
    Works across minor API variations by checking attributes.
    """
    # Most common in nuPlan: lane.baseline_path.discrete_path where each point has x,y
    if hasattr(lane_obj, "baseline_path") and hasattr(lane_obj.baseline_path, "discrete_path"):
        pts = lane_obj.baseline_path.discrete_path
        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
        return xy

    # Fallback: try "centerline" style fields if present
    for attr in ("centerline", "discrete_path", "path"):
        if hasattr(lane_obj, attr):
            pts = getattr(lane_obj, attr)
            try:
                xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
                if len(xy) > 0:
                    return xy
            except Exception:
                pass

    return None


def _outgoing_lane_ids(lane_obj) -> List[str]:
    """
    Get outgoing edges (lane ids). Handles minor API differences.
    """
    for attr in ("outgoing_edges", "outgoing_edges_ids", "outgoing_lane_ids"):
        if hasattr(lane_obj, attr):
            v = getattr(lane_obj, attr)
            if v is None:
                return []
            # Could be list[str] or list[MapObject] or similar
            if len(v) == 0:
                return []
            if isinstance(v[0], str):
                return list(v)
            # If elements are map objects with id
            if hasattr(v[0], "id"):
                return [e.id for e in v]
    return []


def extract_reference_line_xy(
    scenario,
    horizon_lanes: int = 20,
    proximal_radius_m: float = 3.0,
) -> np.ndarray:
    """
    "Reference line" = concatenation of lane centerlines by following outgoing edges
    starting from the lane nearest to ego at iteration 0.
    """
    map_api = scenario.map_api
    ego_state = scenario.get_ego_state_at_iteration(0)
    ego_point = ego_state.rear_axle.point  # Point2D-like with x,y
    ego_xy = np.array([ego_point.x, ego_point.y], dtype=np.float64)

    # Query nearby lanes
    proximal: Dict[SemanticMapLayer, List[object]] = map_api.get_proximal_map_objects(
        point=ego_point,
        radius=proximal_radius_m,
        layers=[SemanticMapLayer.LANE],
    )
    lanes = proximal.get(SemanticMapLayer.LANE, [])
    if not lanes:
        raise RuntimeError(f"No lanes found within {proximal_radius_m}m for scenario token={scenario.token}")

    # Pick best lane by distance to its centerline
    best_lane = None
    best_dist = float("inf")
    for lane in lanes:
        xy = _lane_centerline_xy(lane)
        if xy is None or len(xy) < 2:
            continue
        d = _min_dist_point_to_polyline(ego_xy, xy)
        if d < best_dist:
            best_dist = d
            best_lane = lane

    if best_lane is None:
        raise RuntimeError(f"Could not select a valid lane for scenario token={scenario.token}")

    # Walk lane graph forward
    route_lanes = [best_lane]
    current = best_lane
    for _ in range(horizon_lanes - 1):
        out_ids = _outgoing_lane_ids(current)
        if not out_ids:
            break
        # Choose the first outgoing by default; optionally you can choose based on heading/goal.
        next_lane = map_api.get_map_object(out_ids[0], SemanticMapLayer.LANE)
        if next_lane is None:
            break
        route_lanes.append(next_lane)
        current = next_lane

    # Concatenate centerlines
    ref = []
    for lane in route_lanes:
        xy = _lane_centerline_xy(lane)
        if xy is None or len(xy) == 0:
            continue
        if len(ref) > 0:
            # avoid duplicating the first point if it is identical/very close to previous end
            if np.linalg.norm(xy[0] - ref[-1]) < 1e-3:
                xy = xy[1:]
        ref.extend(xy.tolist())

    if len(ref) < 2:
        raise RuntimeError(f"Reference line too short for scenario token={scenario.token}")

    return np.array(ref, dtype=np.float64)


def compute_reference_line_stats(scenario, horizon_lanes: int, proximal_radius_m: float) -> ReferenceLineResult:
    ref = extract_reference_line_xy(
        scenario=scenario,
        horizon_lanes=horizon_lanes,
        proximal_radius_m=proximal_radius_m,
    )

    curv = polyline_curvature(ref)
    return ReferenceLineResult(
        token=str(scenario.token),
        log_name=str(getattr(scenario, "log_name", "")),
        map_name=str(getattr(scenario, "map_name", "")),
        scenario_type=str(getattr(scenario, "scenario_type", "")),
        start_x=float(ref[0, 0]),
        start_y=float(ref[0, 1]),
        end_x=float(ref[-1, 0]),
        end_y=float(ref[-1, 1]),
        num_samples=int(len(ref)),
        length_m=float(polyline_length(ref)),
        curvature_mean=float(curv.mean()) if len(curv) else 0.0,
        curvature_max=float(curv.max()) if len(curv) else 0.0,
    )


# ----------------------------
# Main / CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="mini", choices=["mini", "trainval"], help="Dataset split under nuplan-v1.1/splits/")
    p.add_argument("--dataset", default="nuplan-v1.1", help="Dataset folder name under NUPLAN_DATA_ROOT (default: nuplan-v1.1)")
    p.add_argument("--map-version", default="nuplan-maps-v1.0", help="Map version string used by devkit")
    p.add_argument("--include-cameras", action="store_true", help="If set, will use sensor_blobs root and enable cameras")
    p.add_argument("--max-scenarios", type=int, default=200, help="Limit total scenarios (for quick stats)")
    p.add_argument("--scenario-types", type=str, default="", help="Comma-separated scenario types, e.g. 'starting_unprotected_cross_turn'")
    p.add_argument("--scenario-tokens-file", type=str, default="", help="Path to a txt/json file containing scenario tokens (one per line or JSON list)")
    p.add_argument("--map-names", type=str, default="", help="Comma-separated map names to filter, e.g. 'us-nv-las-vegas-strip'")
    p.add_argument("--horizon-lanes", type=int, default=20, help="How many lanes to traverse forward for reference line")
    p.add_argument("--proximal-radius-m", type=float, default=3.0, help="Radius for initial lane query around ego")
    p.add_argument("--output-csv", type=str, default="reference_line_stats.csv", help="Output CSV path")
    p.add_argument("--output-jsonl", type=str, default="", help="Optional JSONL output path")
    return p.parse_args()


def load_tokens_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    txt = p.read_text().strip()
    if not txt:
        return []
    # JSON list
    if txt.lstrip().startswith("["):
        arr = json.loads(txt)
        return [str(t).lower() for t in arr]
    # one token per line
    return [line.strip().lower() for line in txt.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()

    data_root = Path(_require_env("NUPLAN_DATA_ROOT")).expanduser().resolve()
    maps_root = Path(_require_env("NUPLAN_MAPS_ROOT")).expanduser().resolve()

    split_dir = _resolve_split_dir(data_root, split=args.split, dataset=args.dataset)
    sensor_root = _resolve_sensor_root(data_root, dataset=args.dataset)

    # Filters
    scenario_types = [s.strip() for s in args.scenario_types.split(",") if s.strip()] or None
    map_names = [s.strip() for s in args.map_names.split(",") if s.strip()] or None
    scenario_tokens = load_tokens_file(args.scenario_tokens_file) if args.scenario_tokens_file else None

    scenario_filter = ScenarioFilter(
        scenario_types=scenario_types,
        scenario_tokens=scenario_tokens,
        log_names=None,
        map_names=map_names,
        num_scenarios_per_type=None,
        limit_total_scenarios=args.max_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
    )

    # Builder
    builder = NuPlanScenarioBuilder(
        data_root=str(split_dir),
        map_root=str(maps_root),
        sensor_root=str(sensor_root),
        db_files=None,  # let devkit discover *.db under data_root
        map_version=args.map_version,
        include_cameras=bool(args.include_cameras),
        max_workers=1,  # leave at 1 for stability; scale up later if needed
        verbose=False,
        scenario_mapping=None,
        vehicle_parameters=get_pacifica_parameters(),
    )

    worker = SingleMachineParallelExecutor()
    scenarios = builder.get_scenarios(scenario_filter=scenario_filter, worker=worker)

    if not scenarios:
        raise RuntimeError("No scenarios matched your filter. Relax filters or check dataset paths.")

    rows: List[ReferenceLineResult] = []
    failures = 0

    for s in scenarios:
        try:
            rows.append(compute_reference_line_stats(s, horizon_lanes=args.horizon_lanes, proximal_radius_m=args.proximal_radius_m))
        except Exception as e:
            failures += 1
            # Keep going; print short info
            print(f"[WARN] token={getattr(s, 'token', '')} failed: {type(e).__name__}: {e}", file=sys.stderr)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(args.output_csv, index=False)

    if args.output_jsonl:
        outp = Path(args.output_jsonl)
        with outp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r.__dict__) + "\n")

    # Console summary
    print(f"Scenarios processed: {len(scenarios)}")
    print(f"Successful: {len(rows)}")
    print(f"Failed: {failures}")
    if len(df):
        print("\nKey stats:")
        print(df[["length_m", "num_samples", "curvature_mean", "curvature_max"]].describe(percentiles=[0.1, 0.5, 0.9]))

    print(f"\nWrote: {args.output_csv}")
    if args.output_jsonl:
        print(f"Wrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()

"""
----------------------------
How to use
----------------------------

1) Set environment variables (example paths):
   export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
   export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"

Per devkit docs, NUPLAN_DATA_ROOT is typically the dataset root, not the split folder.

2) Run on mini (quick):
   python nuplan_reference_line_stats.py --split mini --max-scenarios 200 --output-csv mini_ref_stats.csv

3) Run on trainval with a specific map:
   python nuplan_reference_line_stats.py --split trainval --map-names us-nv-las-vegas-strip --max-scenarios 1000

4) Run on specific scenario tokens:
   # tokens.txt: one token per line
   python nuplan_reference_line_stats.py --split trainval --scenario-tokens-file tokens.txt --max-scenarios 999999

Notes:
- If your NUPLAN_DATA_ROOT already points to .../nuplan-v1.1/splits/trainval (or mini), the script will accept it.
- If reference line definition must match PLUTO exactly, you may want to change the lane-walk logic to use the scenario's
  route plan (if available in your pipeline). This script uses lane-graph follow from the nearest lane to ego at t=0,
  which is usually what supervisors mean by “reference line” in map terms.
"""
