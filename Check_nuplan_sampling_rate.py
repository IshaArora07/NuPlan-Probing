#!/usr/bin/env python3
"""
Check nuPlan scenario sample rate and iteration count.

Verifies what sample_interval nuPlan actually uses internally,
so we can match it in the precompute script.

Usage:
python check_nuplan_sample_rate.py --split mini --n_scenarios 5
"""

import os
import argparse
from pathlib import Path

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import (
    SingleMachineParallelExecutor,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--n_scenarios", type=int, default=5)

    args = parser.parse_args()

    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / args.split

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=1,
    )

    scenarios = builder.get_scenarios(
        ScenarioFilter(limit_total_scenarios=args.n_scenarios),
        SingleMachineParallelExecutor(use_process_pool=False, num_workers=1),
    )

    print("\n" + "─" * 100)
    print(
        f"  {'token':<16}  {'iterations':>10}  {'dt (s)':>8}  "
        f"{'total (s)':>10}  "
        f"{'hist_steps @0.05s':>18}  "
        f"{'hist_steps @0.1s':>17}  "
        f"{'hist_steps @dt':>15}"
    )
    print("─" * 100)

    for s in scenarios[: args.n_scenarios]:

        n = s.get_number_of_iterations()
        dt = s.database_interval

        total = (n - 1) * dt if dt > 0 else 0.0

        hist_005 = int(round(2.0 / 0.05))  # 20Hz
        hist_01 = int(round(2.0 / 0.1))  # 10Hz
        hist_dt = int(round(2.0 / dt)) if dt > 0 else "?"

        print(
            f"  {s.token[:16]:<16}  {n:>10}  {dt:>8.4f}  "
            f"{total:>10.1f}  "
            f"{hist_005:>18}  "
            f"{hist_01:>17}  "
            f"{str(hist_dt):>15}"
        )

    print("─" * 100)

    print(
        """
Key questions:

1. What is dt?
   0.05s (20Hz) → history_steps = 40, future_steps = 160
   0.10s (10Hz) → history_steps = 20, future_steps = 80

2. Does total duration match expected?
   Expected = history(2s) + future(8s) = 10s total
   If total >> 10s → scenario includes more than history+future window

3. Do history_steps @dt match PlutoFeatureBuilder?
   PlutoFeatureBuilder default: sample_interval=0.1 → history_steps=20
   Precompute default:          sample_interval=0.05 → history_steps=40

   These MUST match for correct present-frame alignment.

► Run precompute with --sample_interval matching dt shown above.
"""
    )


if __name__ == "__main__":
    main()
