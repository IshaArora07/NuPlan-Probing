#!/usr/bin/env python3
"""
Targeted check: find exactly why compute_ego_xyh_8s only covers ~4s.

Simulates exactly what compute_ego_xyh_8s does and compares against
what get_ego_future_trajectory returns.

Usage:
python debug_precompute_horizon.py --split mini
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_ego(gx, gy, ox, oy, oh):
    dx, dy = gx - ox, gy - oy
    c, s = math.cos(-oh), math.sin(-oh)
    return c * dx - s * dy, s * dx + c * dy


def path_len(xs, ys):
    dx = np.diff(np.array(xs))
    dy = np.diff(np.array(ys))
    return float(np.sum(np.hypot(dx, dy)))


def straight_dist(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))


def simulate_compute_ego_xyh_8s(
    scenario,
    future_horizon,
    sample_interval,
    history_horizon=0.0,
):
    """
    Simulate exactly what compute_ego_xyh_8s does.

    history_horizon=0.0 → start from iteration 0 (original)
    history_horizon=2.0 → start from history_steps (after offset fix)
    """

    max_iter = scenario.get_number_of_iterations()
    future_steps = int(round(future_horizon / sample_interval))
    hist_steps = int(round(history_horizon / sample_interval))

    start_iter = min(hist_steps, max_iter - 1)
    end_iter = min(hist_steps + future_steps + 1, max_iter)
    n_iters = end_iter - start_iter

    xs, ys, hs = [], [], []

    for i in range(start_iter, end_iter):
        ego = scenario.get_ego_state_at_iteration(i)
        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
        hs.append(float(ego.rear_axle.heading))

    return xs, ys, hs, {
        "max_iter": max_iter,
        "hist_steps": hist_steps,
        "future_steps": future_steps,
        "start_iter": start_iter,
        "end_iter": end_iter,
        "n_iters_used": n_iters,
        "duration_s": (n_iters - 1) * sample_interval,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--n_scenarios", type=int, default=3)
    parser.add_argument("--future_horizon", type=float, default=8.0)
    parser.add_argument(
        "--history_horizon",
        type=float,
        default=0.0,
        help="Set to 2.0 if you applied the history offset fix",
    )
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=0.05,
        help="What your precompute script uses",
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

    pluto_si = 0.1
    pluto_fut_samps = int(args.future_horizon / pluto_si)

    print("\nSettings under test:")
    print(f"  precompute sample_interval = {args.sample_interval}s")
    print(f"  precompute history_horizon = {args.history_horizon}s")
    print(f"  precompute future_horizon  = {args.future_horizon}s")
    print(f"  PlutoFeatureBuilder si     = {pluto_si}s future_samples={pluto_fut_samps}")

    for s in scenarios[: args.n_scenarios]:
        print(f"\n{'='*70}")
        print(
            f"token = {s.token[:20]}  "
            f"n_iters={s.get_number_of_iterations()}  "
            f"dt={s.database_interval}s"
        )
        print(f"{'='*70}")

        xs, ys, hs, info = simulate_compute_ego_xyh_8s(
            s,
            future_horizon=args.future_horizon,
            sample_interval=args.sample_interval,
            history_horizon=args.history_horizon,
        )

        pl = path_len(xs, ys)
        sd = straight_dist(xs, ys)

        dh = 0.0
        if len(hs) >= 2:
            dh = math.degrees(wrap_to_pi(hs[-1] - hs[0]))

        if len(xs) >= 2:
            ex, ey = to_ego(xs[-1], ys[-1], xs[0], ys[0], hs[0])
        else:
            ex, ey = 0.0, 0.0

        print("\n  [PRECOMPUTE simulate_compute_ego_xyh_8s]")
        for k, v in info.items():
            print(f"  {k:<12} = {v}")

        print(f"  path_length   = {pl:.2f}m")
        print(f"  straight_dist = {sd:.2f}m")
        print(f"  delta_heading = {dh:+.2f}°")
        print(f"  ego endpoint  = x={ex:+.2f}  y={ey:+.2f}")

        if abs(info["duration_s"] - args.future_horizon) > 0.5:
            print(f"\n  ✗ DURATION MISMATCH: {info['duration_s']:.2f}s vs {args.future_horizon}s")

            if info["end_iter"] == info["max_iter"]:
                print("    → hit max_iter → scenario exhausted")
        else:
            print("\n  ✓ duration matches")

        # Pluto trajectory
        future = list(
            s.get_ego_future_trajectory(
                iteration=0,
                time_horizon=args.future_horizon,
                num_samples=pluto_fut_samps,
            )
        )

        if future:
            f0 = future[0].rear_axle
            fN = future[-1].rear_axle

            ex_pluto, ey_pluto = to_ego(
                fN.x,
                fN.y,
                xs[0] if xs else f0.x,
                ys[0] if ys else f0.y,
                hs[0] if hs else f0.heading,
            )

            print("\n  [PLUTO]")
            print(f"  samples={len(future)} duration={len(future)*pluto_si:.1f}s")
            print(f"  ego endpoint = x={ex_pluto:+.2f} y={ey_pluto:+.2f}")

            print("\n  ── comparison ──")
            print(f"  precompute : x={ex:+.2f} y={ey:+.2f}")
            print(f"  pluto      : x={ex_pluto:+.2f} y={ey_pluto:+.2f}")

            if abs(ex - ex_pluto) < 5.0:
                print("  ✓ roughly aligned")
            else:
                print("  ✗ mismatch")


if __name__ == "__main__":
    main()
