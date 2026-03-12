#!/usr/bin/env python3
"""
Verify what iteration 0 corresponds to in nuPlan scenarios,
and compare it against what PlutoFeatureBuilder uses as present.

This confirms whether iteration 0 = present frame (initial_ego_state)
or whether there is an offset.

Usage:
python verify_iteration_zero.py --split mini --n_scenarios 3
"""

import os
import argparse
from pathlib import Path

import numpy as np

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
    parser.add_argument("--n_scenarios", type=int, default=3)
    parser.add_argument("--history_horizon", type=float, default=2.0)
    parser.add_argument("--future_horizon", type=float, default=8.0)
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=0.1,
        help="PlutoFeatureBuilder sample_interval (default 0.1)",
    )

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

    history_samples = int(args.history_horizon / args.sample_interval)
    future_samples = int(args.future_horizon / args.sample_interval)

    dt_raw = 0.05  # nuPlan raw rate

    print("\nPlutoFeatureBuilder settings:")
    print(f"  sample_interval  = {args.sample_interval}s")
    print(f"  history_samples  = {history_samples}")
    print(f"  future_samples   = {future_samples}")
    print(f"  total samples    = {history_samples + 1 + future_samples}")

    for s in scenarios[: args.n_scenarios]:

        n_iters = s.get_number_of_iterations()

        print("\n" + "=" * 70)
        print(
            f"token = {s.token}  total_iters={n_iters}  "
            f"total_dur={(n_iters - 1) * dt_raw:.1f}s"
        )
        print("=" * 70)

        # iteration 0 state
        iter0_state = s.get_ego_state_at_iteration(0)
        initial_state = s.initial_ego_state

        iter0_xy = iter0_state.rear_axle.array
        init_xy = initial_state.rear_axle.array

        iter0_h = iter0_state.rear_axle.heading
        init_h = initial_state.rear_axle.heading

        pos_match = np.allclose(iter0_xy, init_xy, atol=1e-3)
        h_match = abs(iter0_h - init_h) < 1e-4

        print("\n  iteration 0 vs initial_ego_state:")
        print(
            f"  iter0   xy=({iter0_xy[0]:.4f}, {iter0_xy[1]:.4f})  "
            f"h={iter0_h:.6f}"
        )
        print(
            f"  initial xy=({init_xy[0]:.4f}, {init_xy[1]:.4f})  "
            f"h={init_h:.6f}"
        )

        print(f"  position match : {'✓ YES' if pos_match else '✗ NO'}")
        print(f"  heading  match : {'✓ YES' if h_match else '✗ NO'}")

        if pos_match and h_match:
            print("  → iteration 0 IS the present frame (initial_ego_state)")
        else:

            offset_iters = None

            for i in range(1, min(100, n_iters)):

                st = s.get_ego_state_at_iteration(i)

                if np.allclose(st.rear_axle.array, init_xy, atol=1e-3):
                    offset_iters = i
                    break

            if offset_iters is not None:

                print(
                    f"  → initial_ego_state found at iteration {offset_iters} "
                    f"= {offset_iters * dt_raw:.2f}s into scenario"
                )

                print(
                    f"  → iteration 0 is {offset_iters * dt_raw:.2f}s BEFORE present"
                )

            else:

                print("  → could not find initial_ego_state in first 100 iterations")

        # Past trajectory
        print(
            f"\n  get_ego_past_trajectory(time_horizon={args.history_horizon}, "
            f"num_samples={history_samples}):"
        )

        past = list(
            s.get_ego_past_trajectory(
                iteration=0,
                time_horizon=args.history_horizon,
                num_samples=history_samples,
            )
        )

        print(f"  returned {len(past)} past states")

        if past:
            p0 = past[0].rear_axle
            pN = past[-1].rear_axle

            print(
                f"  past[0]  xy=({p0.x:.4f}, {p0.y:.4f})  "
                f"h={p0.heading:.6f}"
            )

            print(
                f"  past[-1] xy=({pN.x:.4f}, {pN.y:.4f})  "
                f"h={pN.heading:.6f}"
            )

        # Future trajectory
        print(
            f"\n  get_ego_future_trajectory(time_horizon={args.future_horizon}, "
            f"num_samples={future_samples}):"
        )

        future = list(
            s.get_ego_future_trajectory(
                iteration=0,
                time_horizon=args.future_horizon,
                num_samples=future_samples,
            )
        )

        print(f"  returned {len(future)} future states")

        if future:
            f0 = future[0].rear_axle
            fN = future[-1].rear_axle

            print(
                f"  future[0]  xy=({f0.x:.4f}, {f0.y:.4f})  "
                f"h={f0.heading:.6f}"
            )

            print(
                f"  future[-1] xy=({fN.x:.4f}, {fN.y:.4f})  "
                f"h={fN.heading:.6f}"
            )

        # 8s endpoint comparison
        iter_8s_20hz = int(round(args.future_horizon / dt_raw))

        print("\n  8s future endpoint comparison:")

        if iter_8s_20hz < n_iters:

            st = s.get_ego_state_at_iteration(iter_8s_20hz)

            print(
                f"  via iteration {iter_8s_20hz} (20Hz): "
                f"xy=({st.rear_axle.x:.4f}, {st.rear_axle.y:.4f}) "
                f"h={st.rear_axle.heading:.6f}"
            )

    print("\n" + "=" * 70)
    print(
        """
Summary of what to look for:

1. iteration 0 == initial_ego_state
   ✓ YES → present frame is correct
   ✗ NO  → offset exists

2. 20Hz vs 10Hz endpoint direction agrees
   ✓ YES → resampling not the issue
   ✗ NO  → resampling may shift borderline cases across y=0
"""
    )


if __name__ == "__main__":
    main()
