#!/usr/bin/env python3
"""
verify_pudo_layers_in_gpkg.py

Small, fast verification script:
- Takes either:
    (A) a nuPlan scenario DB folder path (e.g. .../splits/trainval1)
        -> loads just ONE scenario, grabs scenario.map_api, then inspects the underlying GPKG vector layers
    OR
    (B) a map location name directly (e.g. us-nv-las-vegas-strip)
        -> loads GPKGMapsDB directly and prints vector layer names

Goal: confirm whether PUDO / EXTENDED_PUDO (or any pickup/dropoff-like layer) exists in your map package.

Usage examples:

1) Using a DB folder (recommended, matches your exact runtime objects):
   export NUPLAN_DATA_ROOT=/path/to/nuplan
   export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
   python verify_pudo_layers_in_gpkg.py \
     --db_folder "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval1" \
     --max_scenarios 1

2) Using location name only:
   export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
   python verify_pudo_layers_in_gpkg.py \
     --location us-nv-las-vegas-strip \
     --map_version nuplan-maps-v1.0

What you should paste back to me:
- The printed "PUDO-ish layers found" list
- And whether it successfully loaded a PUDO layer (gdf shape + columns)
"""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NuPlan imports (scenario route)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor

# NuPlan maps_db import (direct route)
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB


# -------------------------------------------------------------------
# Helpers from my earlier message (but made standalone + typed)
# -------------------------------------------------------------------
def _safe_vector_layer_names_from_map_api(map_api) -> List[str]:
    """Return all vector layer names in the gpkg for this map location (best-effort)."""
    try:
        maps_db = getattr(map_api, "_maps_db", None)
        if maps_db is None:
            return []
        location = getattr(map_api, "map_name", None)
        if not location:
            return []
        return list(maps_db.vector_layer_names(location))
    except Exception:
        return []


def _load_vector_layer_by_name_from_map_api(map_api, layer_name: str):
    """Load a vector layer directly from maps_db (best-effort)."""
    try:
        if not layer_name:
            return None
        maps_db = getattr(map_api, "_maps_db", None)
        if maps_db is None:
            return None
        location = getattr(map_api, "map_name", None)
        if not location:
            return None
        return maps_db.load_vector_layer(location, layer_name)
    except Exception:
        return None


def load_pudo_layers_direct_from_map_api(map_api) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Returns (pudo_gdf, ext_pudo_gdf, debug_dict).

    It searches the GPKG’s vector layer names for anything that looks like PUDO / pickup-dropoff.
    If not present, returns (None, None, debug).
    """
    names = _safe_vector_layer_names_from_map_api(map_api)
    upper = [n.upper() for n in names]

    def find_layer(match_terms: List[str]) -> Optional[str]:
        # Exact match (case sensitive OR case-insensitive)
        for cand in match_terms:
            if cand in names:
                return cand
            cu = cand.upper()
            if cu in upper:
                return names[upper.index(cu)]

        # Fuzzy contains (first hit)
        for i, n in enumerate(upper):
            if any(term in n for term in ["PUDO", "PICK_UP", "DROP_OFF", "PICKUP", "DROPOFF"]):
                return names[i]
        return None

    pudo_name = find_layer(["PUDO", "PUDOS", "PICK_UP_DROP_OFF", "PICKUP_DROPOFF", "pick_up_drop_off"])
    ext_name = find_layer(["EXTENDED_PUDO", "EXTENDED_PUDOS", "EXTENDED_PICK_UP_DROP_OFF", "extended_pick_up_drop_off"])

    pudo_gdf = _load_vector_layer_by_name_from_map_api(map_api, pudo_name) if pudo_name else None
    ext_gdf = _load_vector_layer_by_name_from_map_api(map_api, ext_name) if ext_name else None

    return (
        pudo_gdf,
        ext_gdf,
        {
            "map_location": getattr(map_api, "map_name", None),
            "num_vector_layers": len(names),
            "pudo_layer_name": pudo_name,
            "ext_pudo_layer_name": ext_name,
            "pudo_loaded": bool(pudo_gdf is not None),
            "ext_loaded": bool(ext_gdf is not None),
        },
    )


# -------------------------------------------------------------------
# Scenario loading (loads only a tiny number, default 1)
# -------------------------------------------------------------------
def _load_one_scenario_from_db_folder(db_folder: str, max_scenarios: int = 1):
    """
    Loads up to max_scenarios scenarios from a *single* split folder path containing *.db files.
    This avoids scanning 17M: it only reads from that folder and only returns a few scenarios.
    """
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=1)

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=max_scenarios,
    )

    builder = NuPlanScenarioBuilder(
        data_root=str(db_folder),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=1,
    )

    scenarios = builder.get_scenarios(scenario_filter, worker)
    if not scenarios:
        raise RuntimeError(f"No scenarios returned from db_folder={db_folder}")
    return scenarios[:max_scenarios]


# -------------------------------------------------------------------
# Direct GPKG route (no scenarios)
# -------------------------------------------------------------------
def _print_layers_from_location(location: str, map_version: str, map_root: str) -> None:
    maps_db = GPKGMapsDB(map_version=map_version, map_root=map_root)
    names = list(maps_db.vector_layer_names(location))
    hits = [n for n in names if any(k in n.upper() for k in ["PUDO", "PICK", "DROP"])]
    print(f"\n[INFO] location={location}")
    print(f"[INFO] map_version={map_version}")
    print(f"[INFO] num_vector_layers={len(names)}")
    print("[INFO] PUDO-ish layers found:")
    for n in hits:
        print("  -", n)
    if not hits:
        print("  (none)")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_folder", type=str, default="", help="Path to a split folder containing *.db files")
    parser.add_argument("--max_scenarios", type=int, default=1, help="How many scenarios to load (db_folder mode)")
    parser.add_argument("--location", type=str, default="", help="Map location name, e.g. us-nv-las-vegas-strip")
    parser.add_argument("--map_version", type=str, default="nuplan-maps-v1.0", help="Map version string")
    args = parser.parse_args()

    if not args.db_folder and not args.location:
        raise SystemExit("Provide either --db_folder OR --location")

    if args.db_folder:
        db_folder = str(Path(args.db_folder).expanduser().resolve())
        print(f"[INFO] db_folder={db_folder}")
        if not Path(db_folder).exists():
            raise FileNotFoundError(db_folder)

        # Load just 1 scenario (or small number)
        scenarios = _load_one_scenario_from_db_folder(db_folder, max_scenarios=args.max_scenarios)
        sc = scenarios[0]

        token = getattr(sc, "token", None)
        scenario_type = getattr(sc, "scenario_type", None)
        map_api = getattr(sc, "map_api", None)

        print(f"[INFO] sample_token={token}")
        print(f"[INFO] sample_scenario_type={scenario_type}")
        print(f"[INFO] scenario.map_name={getattr(sc, 'map_name', None)}")
        print(f"[INFO] map_api type={type(map_api)}")
        print(f"[INFO] map_api.map_name={getattr(map_api, 'map_name', None)}")
        print(f"[INFO] has map_api._maps_db={hasattr(map_api, '_maps_db')}")

        # List vector layers (via map_api -> maps_db)
        names = _safe_vector_layer_names_from_map_api(map_api)
        hits = [n for n in names if any(k in n.upper() for k in ["PUDO", "PICK", "DROP"])]
        print(f"\n[INFO] num_vector_layers={len(names)}")
        print("[INFO] PUDO-ish layers found:")
        for n in hits:
            print("  -", n)
        if not hits:
            print("  (none)")

        # Try to load PUDO layers
        pudo_gdf, ext_gdf, dbg = load_pudo_layers_direct_from_map_api(map_api)
        print("\n[INFO] load_pudo_layers_direct_from_map_api debug:")
        for k, v in dbg.items():
            print(f"  {k}: {v}")

        def summarize_gdf(name: str, gdf):
            if gdf is None:
                print(f"[INFO] {name}: None")
                return
            try:
                print(f"[INFO] {name}: shape={gdf.shape}")
                cols = list(getattr(gdf, "columns", []))
                print(f"[INFO] {name}: first 25 cols={cols[:25]}")
            except Exception as e:
                print(f"[INFO] {name}: loaded but failed to summarize: {e}")

        summarize_gdf("PUDO_GDF", pudo_gdf)
        summarize_gdf("EXT_PUDO_GDF", ext_gdf)

        return

    # location-only mode
    map_root = os.environ["NUPLAN_MAPS_ROOT"]
    _print_layers_from_location(args.location, args.map_version, map_root)


if __name__ == "__main__":
    main()
