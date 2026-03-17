#!/usr/bin/env python3
"""
EMoE Pipeline Alignment Verifier

Comprehensive check that the precompute script, feature cache, and
trajectory.gz are all aligned in terms of:

1. nuPlan scenario sample rate and iteration count
2. n_iterations available vs future_steps needed
3. travel_distance_m in labels vs trajectory path length
4. anchor endpoint scale vs trajectory endpoint scale
5. ego frame convention (left=y>0, right=y<0)
6. present frame alignment (iteration 0 = initial_ego_state)
7. sample_interval consistency across pipelines
8. class distribution sanity (no empty classes)

Usage:
python verify_emoe_alignment.py \
--cache_dir ./nuplan_cache \
--labels_path ./emoe_precomputed/scene_labels.jsonl \
--anchors_path ./emoe_precomputed/scene_anchors.npy \
--split mini \
--n_scenarios 10 \
--future_horizon 8.0 \
--history_horizon 2.0 \
--sample_interval 0.1
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

# ── colours ───────────────────────────────────────────────────────────────────

GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
BLU = "\033[94m"
RST = "\033[0m"
BOLD = "\033[1m"


def ok(m):
    print(f"{GRN}  ✓ {m}{RST}")


def warn(m):
    print(f"{YLW}  ⚠ {m}{RST}")


def fail(m):
    print(f"{RED}  ✗ {m}{RST}")


def hdr(m):
    print(f"\n{BOLD}{BLU}{'─'*65}\n  {m}\n{'─'*65}{RST}")


def sub(m):
    print(f"    {m}")


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]

PASS = 0
WARN_COUNT = 0
FAIL_COUNT = 0


def record(level, msg):
    global PASS, WARN_COUNT, FAIL_COUNT
    if level == "ok":
        ok(msg)
        PASS += 1
    elif level == "warn":
        warn(msg)
        WARN_COUNT += 1
    elif level == "fail":
        fail(msg)
        FAIL_COUNT += 1


# ── loaders ───────────────────────────────────────────────────────────────────

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
        arr = np.array(raw["data"] if isinstance(raw, dict) else raw)
        return arr
    except Exception:
        return None


def path_length(arr):
    diffs = np.diff(arr[:, :2], axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def to_ego(gx, gy, ox, oy, oh):
    dx, dy = gx - ox, gy - oy
    c, s = math.cos(-oh), math.sin(-oh)
    return c * dx - s * dy, s * dx + c * dy


def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── check 1: nuPlan scenario sample rate ──────────────────────────────────────

def check_sample_rate(scenarios, future_horizon, history_horizon, sample_interval):
    hdr("CHECK 1 — nuPlan scenario sample rate and iteration count")

    # only query the FIRST scenario — avoids slow per-scenario DB calls
    s0 = scenarios[0]
    dt_raw = s0.database_interval
    n_iters = s0.get_number_of_iterations()
    n_iters_median = float(n_iters)
    duration_median = (n_iters - 1) * dt_raw

    sub(f"database_interval (dt_raw)    = {dt_raw:.4f}s  ({1/dt_raw:.0f} Hz)")
    sub(f"median n_iterations           = {n_iters_median:.0f}")
    sub(f"median scenario duration      = {duration_median:.1f}s")
    sub(f"your --sample_interval        = {sample_interval}s  ({1/sample_interval:.0f} Hz)")
    sub(f"your --future_horizon         = {future_horizon}s")
    sub(f"your --history_horizon        = {history_horizon}s")

    future_steps_raw = int(round(future_horizon / dt_raw))
    hist_steps_raw = int(round(history_horizon / dt_raw))
    future_steps_si = int(round(future_horizon / sample_interval))
    hist_steps_si = int(round(history_horizon / sample_interval))

    sub(
        f"\n    at dt_raw={dt_raw}s:  history_steps={hist_steps_raw}  "
        f"future_steps={future_steps_raw}  total={hist_steps_raw + future_steps_raw + 1}"
    )
    sub(
        f"    at sample_interval={sample_interval}s:  history_steps={hist_steps_si}  "
        f"future_steps={future_steps_si}  total={hist_steps_si + future_steps_si + 1}"
    )

    # critical check: does n_iters cover future_steps at dt_raw?
    min_iters = n_iters
    if min_iters >= future_steps_raw + 1:
        record(
            "ok",
            f"n_iterations ({min_iters}) >= future_steps_at_dt_raw "
            f"({future_steps_raw}) — full 8s future available at raw rate",
        )
    else:
        record(
            "fail",
            f"n_iterations ({min_iters}) < future_steps_at_dt_raw "
            f"({future_steps_raw}) — precompute only covers "
            f"{(min_iters - 1) * dt_raw:.1f}s not {future_horizon}s!",
        )
        sub(
            f"    → precompute --sample_interval should be {sample_interval}s "
            f"(future_steps={future_steps_si} fits in {min_iters} iters)"
        )

    if min_iters >= future_steps_si + 1:
        record(
            "ok",
            f"n_iterations ({min_iters}) >= future_steps_at_sample_interval "
            f"({future_steps_si})",
        )
    else:
        record(
            "warn",
            f"n_iterations ({min_iters}) < future_steps_at_sample_interval "
            f"({future_steps_si})",
        )

    return dt_raw, future_steps_raw, future_steps_si, hist_steps_raw, hist_steps_si


# ── check 2: present frame alignment ─────────────────────────────────────────

def check_present_frame(scenarios):
    hdr("CHECK 2 — Present frame alignment (iteration 0 = initial_ego_state)")

    # only check 3 scenarios — DB query per scenario is slow
    n_checked = min(3, len(scenarios))
    mismatches = 0
    for s in scenarios[:n_checked]:
        iter0 = s.get_ego_state_at_iteration(0)
        initial = s.initial_ego_state
        pos_ok = np.allclose(iter0.rear_axle.array, initial.rear_axle.array, atol=1e-3)
        h_ok = abs(iter0.rear_axle.heading - initial.rear_axle.heading) < 1e-4
        if not (pos_ok and h_ok):
            mismatches += 1
            sub(f"token={s.token[:16]}  pos_match={pos_ok}  h_match={h_ok}")

    if mismatches == 0:
        record("ok", f"{n_checked} scenarios checked: iteration 0 == initial_ego_state")
    else:
        record(
            "fail",
            f"{mismatches}/{n_checked} scenarios: "
            f"iteration 0 ≠ initial_ego_state — history offset needed",
        )


# ── check 3: trajectory.gz format ─────────────────────────────────────────────

def check_trajectory_format(cache_dir, label_map, n_check=20):
    hdr("CHECK 3 — trajectory.gz format (shape, scale, absolute vs delta)")

    found = 0
    shapes = Counter()
    speeds = []
    first_pts = []
    path_lengths = []

    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue
        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue
            for tok_dir in sorted(tag.iterdir()):
                traj_p = tok_dir / "trajectory.gz"
                if not traj_p.exists():
                    continue
                arr = load_traj(traj_p)
                if arr is None:
                    continue
                shapes[arr.shape] += 1
                pl = path_length(arr)
                path_lengths.append(pl)
                avg_speed = pl / arr.shape[0]
                speeds.append(avg_speed)
                first_pts.append(arr[0, :2].tolist())
                found += 1
                if found >= n_check:
                    break
            if found >= n_check:
                break
        if found >= n_check:
            break

    if not speeds:
        record("fail", "No trajectory.gz files found")
        return

    sub(f"Checked {found} trajectory files")
    sub(f"Shapes found: {dict(shapes)}")
    sub(
        f"Avg speed  : min={min(speeds):.2f}  median={np.median(speeds):.2f}  "
        f"max={max(speeds):.2f} m/s"
    )
    sub(
        f"Path length: min={min(path_lengths):.1f}  median={np.median(path_lengths):.1f}  "
        f"max={max(path_lengths):.1f} m"
    )

    first_arr = np.array(first_pts)
    near_zero = np.all(np.abs(first_arr) < 1.0, axis=1).mean()
    sub(f"First waypoint near (0,0): {near_zero*100:.0f}% of trajectories")

    if near_zero > 0.8:
        record("warn", "Most trajectories start near (0,0) — may be delta/relative format")
        sub("    → anchors use absolute ego-frame; ensure comparison is consistent")
    else:
        record("ok", "Trajectories appear to be absolute ego-frame positions")

    med_speed = float(np.median(speeds))
    if 2.0 < med_speed < 20.0:
        record("ok", f"Median speed {med_speed:.1f} m/s is plausible urban driving")
    else:
        record("fail", f"Median speed {med_speed:.1f} m/s is implausible — check trajectory format")

    return float(np.median(path_lengths))


# ── check 4: anchor vs trajectory scale ───────────────────────────────────────

def check_anchor_scale(cache_dir, label_map, anchors, n_check=30):
    hdr("CHECK 4 — Anchor endpoint scale vs trajectory endpoint scale")

    ratios = defaultdict(list)
    traj_ends = defaultdict(list)
    anch_ends = defaultdict(list)

    found = 0
    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue
        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue
            for tok_dir in sorted(tag.iterdir()):
                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"
                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)
                if cid is None or cid >= 6:
                    continue

                arr = load_traj(traj_p)
                if arr is None:
                    continue

                tok = tok_dir.name
                rec = label_map.get(tok, {})
                ep = rec.get("endpoint_xy")

                tx = float(arr[-1, 0])
                ty = float(arr[-1, 1])
                traj_ends[cid].append([tx, ty])

                if ep:
                    ax, ay = float(ep[0]), float(ep[1])
                    anch_ends[cid].append([ax, ay])
                    if abs(ax) > 1.0:
                        ratios[cid].append(tx / ax)

                found += 1
                if found >= n_check:
                    break
            if found >= n_check:
                break
        if found >= n_check:
            break

    sub(f"Checked {found} tokens across all classes\n")

    all_ok = True
    for c in range(6):
        te = np.array(traj_ends[c]) if traj_ends[c] else None
        ae = np.array(anch_ends[c]) if anch_ends[c] else None

        if te is None:
            sub(f"class {c} ({EMOE_SCENE_TYPES[c]:<28s}): no trajectory data")
            continue

        traj_med_x = float(np.median(np.abs(te[:, 0])))
        traj_med_y = float(np.median(np.abs(te[:, 1])))

        if anchors is not None and c < anchors.shape[0]:
            anc_med_x = float(np.median(np.abs(anchors[c, :, 0])))
            anc_med_y = float(np.median(np.abs(anchors[c, :, 1])))
            scale_x = traj_med_x / anc_med_x if anc_med_x > 0.5 else float("nan")
            scale_y = traj_med_y / anc_med_y if anc_med_y > 0.5 else float("nan")
            sub(
                f"class {c} ({EMOE_SCENE_TYPES[c]:<28s}):  "
                f"traj_med=({traj_med_x:.1f},{traj_med_y:.1f})  "
                f"anch_med=({anc_med_x:.1f},{anc_med_y:.1f})  "
                f"scale_x={scale_x:.2f}  scale_y={scale_y:.2f}"
            )

            if not math.isnan(scale_x) and not (0.7 < scale_x < 1.4):
                all_ok = False
                fail(
                    f"class {c}: x scale mismatch ({scale_x:.2f}x) — "
                    f"anchors and trajectories cover different time horizons"
                )
        else:
            sub(
                f"class {c} ({EMOE_SCENE_TYPES[c]:<28s}):  "
                f"traj_med=({traj_med_x:.1f},{traj_med_y:.1f})  (no anchor npy)"
            )

    if all_ok:
        record("ok", "Anchor and trajectory scales are consistent across all classes")


# ── check 5: label travel_distance vs trajectory path length ──────────────────

def check_travel_distance(cache_dir, label_map, n_check=30):
    hdr("CHECK 5 — Label travel_distance_m vs trajectory path length")

    ratios = []
    examples = []

    found = 0
    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue
        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue
            for tok_dir in sorted(tag.iterdir()):
                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"
                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)
                if cid is None:
                    continue

                arr = load_traj(traj_p)
                if arr is None:
                    continue

                tok = tok_dir.name
                rec = label_map.get(tok)
                if rec is None:
                    continue

                label_dist = rec.get("travel_distance_m", 0.0)
                traj_pl = path_length(arr)

                if label_dist > 1.0:
                    ratio = traj_pl / label_dist
                    ratios.append(ratio)
                    examples.append((tok[:16], label_dist, traj_pl, ratio))

                found += 1
                if found >= n_check:
                    break
            if found >= n_check:
                break
        if found >= n_check:
            break

    if not ratios:
        record("warn", "No matching tokens found between cache and labels")
        return

    med_ratio = float(np.median(ratios))
    sub(f"Checked {len(ratios)} matched tokens")
    sub("traj_path_length / label_travel_dist:")
    sub(f"  median={med_ratio:.3f}  min={min(ratios):.3f}  max={max(ratios):.3f}")
    sub("\n  Examples:")
    sub(f"  {'token':<18}  {'label_dist':>10}  {'traj_pl':>8}  {'ratio':>6}")
    for tok, ld, tp, r in examples[:8]:
        sub(f"  {tok:<18}  {ld:>10.2f}m  {tp:>8.2f}m  {r:>6.2f}x")

    if 0.85 < med_ratio < 1.15:
        record("ok", f"Median ratio {med_ratio:.3f} ≈ 1.0 — scales match")
    elif 1.7 < med_ratio < 2.1:
        record(
            "fail",
            f"Median ratio {med_ratio:.3f} ≈ 2.0 — "
            f"precompute covers ~half the trajectory horizon!\n"
            f"    → precompute used sample_interval too small "
            f"(n_iters exhausted early)\n"
            f"    → fix: use --sample_interval 0.1 in precompute",
        )
    else:
        record("warn", f"Median ratio {med_ratio:.3f} — unexpected scale difference")


# ── check 6: ego frame convention ─────────────────────────────────────────────

def check_ego_frame_convention(cache_dir, label_map, n_check=20):
    hdr("CHECK 6 — Ego frame convention (left=y>0, right=y<0)")

    class_y = defaultdict(list)
    found = 0

    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue
        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue
            for tok_dir in sorted(tag.iterdir()):
                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"
                if not feat_p.exists() or not traj_p.exists():
                    continue

                cid = load_emoe_class(feat_p)
                if cid is None or cid >= 6:
                    continue

                arr = load_traj(traj_p)
                if arr is None:
                    continue

                class_y[cid].append(float(arr[-1, 1]))
                found += 1
                if found >= n_check * 6:
                    break
            if found >= n_check * 6:
                break
        if found >= n_check * 6:
            break

    sub(f"Analysed {found} tokens\n")
    sub(
        f"  {'class':<4}  {'name':<30}  {'n':>4}  "
        f"{'med_y':>8}  {'%y>0':>7}  {'expected':>12}  status"
    )
    sub(f"  {'─'*4}  {'─'*30}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*6}")

    convention_ok = True
    for c in range(6):
        ys = np.array(class_y[c]) if class_y[c] else np.array([])
        if len(ys) == 0:
            sub(f"  {c:<4}  {EMOE_SCENE_TYPES[c]:<30}  {'0':>4}  —")
            continue

        med_y = float(np.median(ys))
        pct_pos = float((ys > 0).mean()) * 100

        if c == 0:
            expected = "y>0 (left)"
            good = med_y > 2.0
        elif c == 2:
            expected = "y<0 (right)"
            good = med_y < -2.0
        elif c in (1, 3):
            expected = "y≈0 (straight)"
            good = abs(med_y) < 5.0
        else:
            expected = "any"
            good = True

        status = f"{GRN}✓{RST}" if good else f"{RED}✗{RST}"
        sub(
            f"  {c:<4}  {EMOE_SCENE_TYPES[c]:<30}  {len(ys):>4}  "
            f"{med_y:>8.2f}  {pct_pos:>6.1f}%  {expected:>12}  {status}"
        )

        if not good:
            convention_ok = False

    if convention_ok:
        record("ok", "Ego frame convention is consistent for all classes")
    else:
        record("fail", "Ego frame convention mismatch — check classifier sign convention or coordinate frame")


# ── check 7: class distribution ───────────────────────────────────────────────

def check_class_distribution(cache_dir, label_map, anchors):
    hdr("CHECK 7 — Class distribution (no empty classes, balanced enough)")

    class_counts = Counter()
    no_label = 0
    no_traj = 0

    for log in sorted(cache_dir.iterdir()):
        if not log.is_dir():
            continue
        for tag in sorted(log.iterdir()):
            if not tag.is_dir():
                continue
            for tok_dir in sorted(tag.iterdir()):
                feat_p = tok_dir / "features.gz"
                traj_p = tok_dir / "trajectory.gz"
                if not traj_p.exists():
                    no_traj += 1
                    continue
                if not feat_p.exists():
                    no_label += 1
                    continue
                cid = load_emoe_class(feat_p)
                if cid is None:
                    no_label += 1
                else:
                    class_counts[cid] += 1

    total_labelled = sum(class_counts.values())
    sub(f"Total tokens with trajectory : {total_labelled + no_label}")
    sub(f"Tokens with emoe label       : {total_labelled}")
    sub(f"Tokens without emoe label    : {no_label}")
    sub(f"Tokens without trajectory    : {no_traj}\n")

    sub(
        f"  {'c':<3}  {'name':<30}  {'n':>6}  {'%':>6}  "
        f"{'anch_std_x':>10}  {'anch_std_y':>10}"
    )
    sub(f"  {'─'*3}  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")

    empty = []
    for c in range(6):
        n = class_counts.get(c, 0)
        pct = 100.0 * n / max(1, total_labelled)
        name = EMOE_SCENE_TYPES[c]

        anch_sx = anch_sy = float("nan")
        if anchors is not None and c < anchors.shape[0]:
            anch_sx = float(np.std(anchors[c, :, 0]))
            anch_sy = float(np.std(anchors[c, :, 1]))

        sub(
            f"  {c:<3}  {name:<30}  {n:>6}  {pct:>5.1f}%  "
            f"{anch_sx:>10.2f}  {anch_sy:>10.2f}"
        )

        if n == 0:
            empty.append(c)

    if empty:
        record("fail", f"Empty classes: {empty} — these experts will receive no training signal")
    else:
        record("ok", "All classes have at least some labelled tokens")

    if no_label > total_labelled:
        record(
            "warn",
            f"More unlabelled ({no_label}) than labelled ({total_labelled}) "
            f"tokens — cache may need rebuilding with updated labels path",
        )
    else:
        record("ok", f"{total_labelled} labelled tokens ready for training")


# ── check 8: anchor zero-init check ───────────────────────────────────────────

def check_anchor_init(anchors):
    hdr("CHECK 8 — Anchor quality (no zero-init, sufficient spread)")

    if anchors is None:
        record("warn", "No anchors_path provided — skipping")
        return

    sub(f"Anchors shape: {anchors.shape}\n")

    for c in range(anchors.shape[0]):
        anc = anchors[c]
        zeros = np.all(anc == 0, axis=1).sum()
        std_x = float(np.std(anc[:, 0]))
        std_y = float(np.std(anc[:, 1]))
        med_x = float(np.median(np.abs(anc[:, 0])))
        med_y = float(np.median(np.abs(anc[:, 1])))
        name = EMOE_SCENE_TYPES[c] if c < len(EMOE_SCENE_TYPES) else f"class_{c}"

        sub(
            f"class {c} ({name:<28s}):  "
            f"zeros={zeros}/{anc.shape[0]}  "
            f"std=({std_x:.2f},{std_y:.2f})  "
            f"med_abs=({med_x:.2f},{med_y:.2f})"
        )

        if zeros == anc.shape[0]:
            fail(f"class {c}: ALL anchors are zero — KMeans had no data for this class")
        elif zeros > 0:
            warn(f"class {c}: {zeros} zero anchors (padding from too few data points)")
        elif std_x < 0.5 and std_y < 0.5:
            warn(f"class {c}: very low anchor spread — anchors may be degenerate")
        else:
            ok(f"class {c}: anchors look healthy")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="mini")
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=3,
        help="Scenarios to load for checks 1+2 only (keep small)",
    )
    parser.add_argument("--future_horizon", type=float, default=8.0)
    parser.add_argument("--history_horizon", type=float, default=2.0)
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=0.1,
        help="PlutoFeatureBuilder sample_interval (default 0.1s = 10Hz)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # load labels
    label_map = {}
    with open(args.labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                label_map[r["token"]] = r
            except Exception:
                continue
    print(f"[INFO] Loaded {len(label_map)} labels from {args.labels_path}")

    # load anchors
    anchors = None
    if args.anchors_path:
        anchors = np.load(args.anchors_path)
        print(f"[INFO] Loaded anchors shape={anchors.shape} from {args.anchors_path}")

    # load nuPlan scenarios
    print(f"[INFO] Loading {args.n_scenarios} nuPlan scenarios from split={args.split}...")
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
    print(f"[INFO] Loaded {len(scenarios)} scenarios\n")

    # ── run all checks ────────────────────────────────────────────────────
    check_sample_rate(
        scenarios, args.future_horizon, args.history_horizon, args.sample_interval
    )
    check_present_frame(scenarios)
    check_trajectory_format(cache_dir, label_map)
    check_anchor_scale(cache_dir, label_map, anchors)
    check_travel_distance(cache_dir, label_map)
    check_ego_frame_convention(cache_dir, label_map)
    check_class_distribution(cache_dir, label_map, anchors)
    check_anchor_init(anchors)

    # ── final summary ─────────────────────────────────────────────────────
    hdr("SUMMARY")
    print(f"  {GRN}PASS : {PASS}{RST}")
    print(f"  {YLW}WARN : {WARN_COUNT}{RST}")
    print(f"  {RED}FAIL : {FAIL_COUNT}{RST}")

    if FAIL_COUNT == 0 and WARN_COUNT == 0:
        print(f"\n  {GRN}{BOLD}All checks passed — pipeline is aligned ✓{RST}")
    elif FAIL_COUNT == 0:
        print(f"\n  {YLW}No failures but {WARN_COUNT} warnings — review above{RST}")
    else:
        print(f"\n  {RED}{BOLD}{FAIL_COUNT} failure(s) detected — fix before training{RST}")
        print(f"\n  Most common fixes:")
        print(f"  1. Scale mismatch → re-run precompute with --sample_interval 0.1")
        print(f"  2. Empty classes  → check _emoe_label_path in PlutoFeatureBuilder and rebuild cache")
        print(f"  3. Frame mismatch → ensure iteration 0 = present frame")


if __name__ == "__main__":
    main()
