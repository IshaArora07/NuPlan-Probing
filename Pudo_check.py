#!/usr/bin/env python3
import os
from pathlib import Path
from collections import Counter

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


def safe_get_map_name(scenario):
    # Try several common places where map name can exist
    for attr in ["map_name", "_map_name"]:
        if hasattr(scenario, attr):
            v = getattr(scenario, attr)
            if isinstance(v, str) and v:
                return v
    # Try through map_api
    m = getattr(scenario, "map_api", None)
    if m is not None and hasattr(m, "map_name"):
        v = getattr(m, "map_name")
        if isinstance(v, str) and v:
            return v
    return None


def main():
    split = os.environ.get("SPLIT", "trainval")
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split
    assert db_root.exists(), f"DB root not found: {db_root}"

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=8,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=1)

    scenarios = builder.get_scenarios(ScenarioFilter(limit_total_scenarios=20), worker)
    print(f"[INFO] loaded {len(scenarios)} scenarios for probing")

    map_names = []
    for s in scenarios:
        mn = safe_get_map_name(s)
        map_names.append(mn)
        print("token:", s.token, "scenario_type:", getattr(s, "scenario_type", None), "map_name:", mn)

    counts = Counter(map_names)
    print("\n[INFO] map_name distribution:", counts)

    # Try listing vector layers for the first non-None map_name we see
    mn0 = next((m for m in map_names if m is not None), None)
    if mn0 is None:
        print("\n[ERROR] map_name is None for all probed scenarios.")
        print("This means your scenario objects are not carrying the map location,")
        print("so any map-db-based layer query (incl PUDO) cannot work until fixed.")
        return

    # Access maps_db through map_api
    mapi = getattr(scenarios[0], "map_api", None)
    if mapi is None or not hasattr(mapi, "_maps_db"):
        print("\n[ERROR] scenario.map_api missing or has no _maps_db")
        return

    maps_db = mapi._maps_db  # GPKGMapsDB
    layer_names = list(maps_db.vector_layer_names(mn0))
    print(f"\n[INFO] vector layers in {mn0}: {len(layer_names)} total")

    pudo_like = [n for n in layer_names if "pudo" in n.lower() or "pick" in n.lower() or "drop" in n.lower()]
    print("\n[INFO] layers containing pudo/pick/drop:")
    for n in sorted(pudo_like):
        print("  ", n)

    # Optional: attempt to load a few likely candidates
    candidates = [
        "pudo", "PUDO", "pudos", "pudo_polygons",
        "extended_pudo", "ext_pudo",
        "pickup_dropoff", "pick_up_drop_off",
        "extended_pickup_dropoff",
    ]
    print("\n[INFO] trying to load candidate layers:")
    for name in candidates:
        try:
            gdf = maps_db.load_vector_layer(mn0, name)
            print(f"  {name:28s} -> len={0 if gdf is None else len(gdf)}")
        except Exception as e:
            print(f"  {name:28s} -> FAIL ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
