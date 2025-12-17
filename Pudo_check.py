#!/usr/bin/env python3
"""
check_pudo_layers.py

Small standalone sanity-check script for nuPlan:
- loads a limited number of scenarios from a split/folder
- prints whether map_api exists
- prints whether PUDO / EXTENDED_PUDO vector layers exist
- prints lengths + column names + detected pudo_type column
- optionally runs a quick "does ego ever get within tol of a PUDO polygon" test

Usage examples:

1) If you are using the official split folder layout:
   export NUPLAN_DATA_ROOT=/path/to/nuplan
   export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps

   python check_pudo_layers.py --split trainval --max_scenarios 20

2) If you are using your custom partition folder that directly contains .db files:
   python check_pudo_layers.py --db_root /path/to/nuplan-v1.1/splits/trainval3 --max_scenarios 20

Optional:
   --check_traversal 1 --tol_m 10.0 --min_hits 1 --sample_step 1
"""

import os
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# Depending on devkit version, the executor import path differs.
# Try both; one should work.
try:
    from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor
except Exception:
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor  # type: ignore

from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def compute_ego_xy(scenario) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def safe_get_vector_layer(map_api, layer: SemanticMapLayer):
    # Same place your classifier uses: map_api._get_vector_map_layer(...)
    try:
        return map_api._get_vector_map_layer(layer)  # type: ignore[attr-defined]
    except Exception:
        return None


def detect_type_col(gdf, candidates: List[str]) -> str:
    if gdf is None:
        return ""
    for c in candidates:
        if c in gdf.columns:
            return c
    return ""


def quick_traversal_check(gdf, xs: np.ndarray, ys: np.ndarray, tol_m: float, min_hits: int, sample_step: int) -> Dict[str, Any]:
    """Very simple: count how many ego samples are within tol of ANY polygon in gdf."""
    if gdf is None or len(gdf) == 0:
        return {"traversed": False, "hits": 0, "min_dist": float("inf")}

    geoms = list(gdf.geometry.values)
    try:
        sindex = gdf.sindex
    except Exception:
        sindex = None

    hits = 0
    min_dist = float("inf")
    tol = float(tol_m)

    for i in range(0, len(xs), max(1, int(sample_step))):
        p = Point(float(xs[i]), float(ys[i]))

        if sindex is not None:
            bbox = (p.x - tol, p.y - tol, p.x + tol, p.y + tol)
            cand_idx = list(sindex.intersection(bbox))
        else:
            cand_idx = list(range(len(geoms)))

        touched = False
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
                    break
                d = float(g.distance(p))
                min_dist = min(min_dist, d)
                if d <= tol:
                    touched = True
                    break
            except Exception:
                continue

        if touched:
            hits += 1
            if hits >= int(min_hits) and min_dist <= tol:
                break

    return {"traversed": (hits >= int(min_hits) and min_dist <= tol), "hits": int(hits), "min_dist": float(min_dist)}


def build_scenarios_from_db_root(db_root: Path, map_root: Path, max_scenarios: int, num_workers: int):
    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=int(num_workers))

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=None if max_scenarios < 0 else int(max_scenarios),
    )

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=int(num_workers),
    )
    scenarios = builder.get_scenarios(scenario_filter, worker)
    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]
    return scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="", help="mini/trainval/... (optional if using --db_root)")
    ap.add_argument("--db_root", type=str, default="", help="Folder that contains the .db files directly")
    ap.add_argument("--max_scenarios", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--check_traversal", type=int, default=0, help="1=also run ego-vs-PUDO distance check")
    ap.add_argument("--tol_m", type=float, default=10.0)
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--sample_step", type=int, default=1)

    args = ap.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    if not map_root:
        raise RuntimeError("Please set NUPLAN_MAPS_ROOT to your nuPlan maps root folder.")
    map_root = Path(map_root)

    if args.db_root:
        db_root = Path(args.db_root).expanduser().resolve()
    else:
        if not data_root:
            raise RuntimeError("Please set NUPLAN_DATA_ROOT or provide --db_root.")
        if not args.split:
            raise RuntimeError("Provide --split (e.g. trainval) or provide --db_root.")
        db_root = Path(data_root) / "nuplan-v1.1" / "splits" / args.split

    if not db_root.exists():
        raise FileNotFoundError(f"db_root does not exist: {db_root}")

    print(f"[INFO] db_root = {db_root}")
    print(f"[INFO] map_root = {map_root}")
    print(f"[INFO] max_scenarios = {args.max_scenarios}")

    scenarios = build_scenarios_from_db_root(db_root, map_root, args.max_scenarios, args.num_workers)
    print(f"[INFO] loaded scenarios: {len(scenarios)}")

    if len(scenarios) == 0:
        print("[WARN] No scenarios loaded. Check db_root path / permissions.")
        return

    for idx, scenario in enumerate(scenarios):
        token = getattr(scenario, "token", "NA")
        map_name = getattr(scenario, "map_name", None)
        map_api = getattr(scenario, "map_api", None)

        print("\n" + "=" * 90)
        print(f"[SCENARIO {idx}] token={token}")
        print(f"map_name={map_name}")
        print(f"has map_api={map_api is not None}")

        if map_api is None:
            continue

        pudo_gdf = safe_get_vector_layer(map_api, SemanticMapLayer.PUDO)
        ext_pudo_gdf = safe_get_vector_layer(map_api, SemanticMapLayer.EXTENDED_PUDO)

        print(f"PUDO layer: {'None' if pudo_gdf is None else 'OK'}  len={0 if pudo_gdf is None else len(pudo_gdf)}")
        print(f"EXT_PUDO layer: {'None' if ext_pudo_gdf is None else 'OK'}  len={0 if ext_pudo_gdf is None else len(ext_pudo_gdf)}")

        if pudo_gdf is not None:
            cols = list(pudo_gdf.columns)
            print("PUDO columns (first 40):", cols[:40])
            pudo_type_col = detect_type_col(pudo_gdf, ["pudo_type_fid", "pudo_type", "type_fid", "type"])
            print("PUDO detected type col:", pudo_type_col)

        if ext_pudo_gdf is not None:
            cols = list(ext_pudo_gdf.columns)
            print("EXT_PUDO columns (first 40):", cols[:40])
            ext_type_col = detect_type_col(ext_pudo_gdf, ["pudo_type_fid", "pudo_type", "type_fid", "type"])
            print("EXT_PUDO detected type col:", ext_type_col)

        if args.check_traversal == 1:
            xs, ys = compute_ego_xy(scenario)
            if len(xs) < 2:
                print("[TRAVERSAL] not enough ego states")
                continue

            t1 = quick_traversal_check(pudo_gdf, xs, ys, args.tol_m, args.min_hits, args.sample_step)
            t2 = quick_traversal_check(ext_pudo_gdf, xs, ys, args.tol_m, args.min_hits, args.sample_step)

            print(f"[TRAVERSAL] PUDO: traversed={t1['traversed']} hits={t1['hits']} min_dist={t1['min_dist']:.3f}")
            print(f"[TRAVERSAL] EXT_PUDO: traversed={t2['traversed']} hits={t2['hits']} min_dist={t2['min_dist']:.3f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
