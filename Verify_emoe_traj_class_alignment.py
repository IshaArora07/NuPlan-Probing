#!/usr/bin/env python3
"""
EMoE Pipeline Alignment Verifier
"""

import os
import gzip
import json
import math
import pickle
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


# ── colours ─────────────────────────────────────────────────────────

GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
BLU = "\033[94m"
RST = "\033[0m"
BOLD = "\033[1m"


def ok(m): print(f"{GRN}  ✓ {m}{RST}")
def warn(m): print(f"{YLW}  ⚠ {m}{RST}")
def fail(m): print(f"{RED}  ✗ {m}{RST}")
def hdr(m): print(f"\n{BOLD}{BLU}{'─'*65}\n  {m}\n{'─'*65}{RST}")
def sub(m): print(f"    {m}")


PASS = 0
WARN_COUNT = 0
FAIL_COUNT = 0


def record(level, msg):
    global PASS, WARN_COUNT, FAIL_COUNT
    if level == "ok":
        ok(msg); PASS += 1
    elif level == "warn":
        warn(msg); WARN_COUNT += 1
    elif level == "fail":
        fail(msg); FAIL_COUNT += 1


# ── loaders ─────────────────────────────────────────────────────────

def load_emoe_class(feat_path):
    try:
        raw = pickle.load(gzip.open(feat_path, "rb"))
        inner = raw["data"]

        if hasattr(inner, "data"):
            inner = inner.data

        if not isinstance(inner, dict):
            return None

        emoe = inner.get("emoe")
        if emoe is None:
            return None

        val = emoe.get("emoe_class_id")
        if val is None:
            return None

        if hasattr(val, "item"):
            val = val.item()

        return int(val)

    except Exception:
        return None


def load_traj(traj_path):
    try:
        raw = pickle.load(gzip.open(traj_path, "rb"))
        return np.array(raw["data"] if isinstance(raw, dict) else raw)
    except Exception:
        return None


def path_length(arr):
    diffs = np.diff(arr[:, :2], axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


# ── check 1 ─────────────────────────────────────────────────────────

def check_sample_rate(scenarios, future_horizon, history_horizon, sample_interval):

    hdr("CHECK 1 — Sample rate and iteration count")

    dt_raw = scenarios[0].database_interval
    n_iters = [s.get_number_of_iterations() for s in scenarios]

    min_iters = min(n_iters)

    future_steps_raw = int(round(future_horizon / dt_raw))
    future_steps_si = int(round(future_horizon / sample_interval))

    sub(f"dt_raw = {dt_raw:.4f}s")
    sub(f"min_iters = {min_iters}")

    if min_iters >= future_steps_raw:
        record("ok", "Enough iterations for raw future horizon")
    else:
        record("fail", "Not enough iterations for raw horizon")

    if min_iters >= future_steps_si:
        record("ok", "Enough iterations for sample_interval horizon")
    else:
        record("warn", "Sample interval may be too small")


# ── check 2 ─────────────────────────────────────────────────────────

def check_present_frame(scenarios):

    hdr("CHECK 2 — Present frame alignment")

    mismatch = 0

    for s in scenarios:
        iter0 = s.get_ego_state_at_iteration(0)
        init = s.initial_ego_state

        if not np.allclose(iter0.rear_axle.array, init.rear_axle.array, atol=1e-3):
            mismatch += 1

    if mismatch == 0:
        record("ok", "Iteration 0 == present")
    else:
        record("fail", f"{mismatch} mismatches found")


# ── check 3 ─────────────────────────────────────────────────────────

def check_trajectory_format(cache_dir):

    hdr("CHECK 3 — trajectory.gz format")

    speeds = []
    count = 0

    for p in cache_dir.rglob("trajectory.gz"):

        arr = load_traj(p)
        if arr is None:
            continue

        pl = path_length(arr)
        speeds.append(pl / arr.shape[0])

        count += 1
        if count >= 20:
            break

    if not speeds:
        record("fail", "No trajectories found")
        return

    med_speed = float(np.median(speeds))

    sub(f"median speed = {med_speed:.2f} m/s")

    if 2 < med_speed < 20:
        record("ok", "Speed plausible")
    else:
        record("fail", "Speed implausible")


# ── check 7 (kept important) ─────────────────────────────────────────

def check_class_distribution(cache_dir):

    hdr("CHECK 7 — Class distribution")

    counts = Counter()

    for feat in cache_dir.rglob("features.gz"):
        cid = load_emoe_class(feat)
        if cid is not None:
            counts[cid] += 1

    for c in range(6):
        sub(f"class {c}: {counts[c]}")

    if any(counts[c] == 0 for c in range(6)):
        record("fail", "Empty classes detected")
    else:
        record("ok", "All classes present")


# ── main ────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument("--n_scenarios", type=int, default=10)
    parser.add_argument("--future_horizon", type=float, default=8.0)
    parser.add_argument("--history_horizon", type=float, default=2.0)
    parser.add_argument("--sample_interval", type=float, default=0.1)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    print("[INFO] Loading scenarios...")

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

    check_sample_rate(
        scenarios,
        args.future_horizon,
        args.history_horizon,
        args.sample_interval,
    )

    check_present_frame(scenarios)
    check_trajectory_format(cache_dir)
    check_class_distribution(cache_dir)

    hdr("SUMMARY")

    print(f"{GRN}PASS: {PASS}{RST}")
    print(f"{YLW}WARN: {WARN_COUNT}{RST}")
    print(f"{RED}FAIL: {FAIL_COUNT}{RST}")


if __name__ == "__main__":
    main()
