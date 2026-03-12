#!/usr/bin/env python3

"""
Verify coordinate frame consistency between:

1. cached trajectory targets (trajectory.gz)
2. classification pipeline endpoints

This script checks:
- coordinate equality
- sign flip possibility
- rotation mismatch
"""

import gzip
import pickle
import json
import argparse
from pathlib import Path
import numpy as np

# ------------------------------------------------
# Load cached trajectory (8s)
# ------------------------------------------------

def load_cached_traj(traj_path):

    with gzip.open(traj_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        arr = obj.get("data", obj)
    else:
        arr = obj

    if hasattr(arr, "numpy"):
        arr = arr.numpy()

    arr = np.asarray(arr)

    return arr[:, :2]   # (x,y)


# ------------------------------------------------
# Compute endpoint from classification pipeline
# ------------------------------------------------

def compute_endpoint_from_scenario(scenario):

    xs, ys, hs = [], [], []

    for i in range(scenario.get_number_of_iterations()):
        ego = scenario.get_ego_state_at_iteration(i)

        xs.append(float(ego.rear_axle.x))
        ys.append(float(ego.rear_axle.y))
        hs.append(float(ego.rear_axle.heading))

    xs = np.array(xs)
    ys = np.array(ys)
    hs = np.array(hs)

    # 8s horizon
    T = min(len(xs)-1, 8)

    x0, y0 = xs[0], ys[0]
    xT, yT = xs[T], ys[T]

    dx = xT - x0
    dy = yT - y0

    theta0 = hs[0]

    c = np.cos(-theta0)
    s = np.sin(-theta0)

    x_rel = c*dx - s*dy
    y_rel = s*dx + c*dy

    return np.array([x_rel, y_rel])


# ------------------------------------------------
# Comparison diagnostics
# ------------------------------------------------

def compare_frames(cached_ep, classifier_ep):

    diff = np.linalg.norm(cached_ep - classifier_ep)

    flipped = cached_ep.copy()
    flipped[1] *= -1

    diff_flip = np.linalg.norm(flipped - classifier_ep)

    return diff, diff_flip


# ------------------------------------------------
# Main verification loop
# ------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--num_samples", type=int, default=10)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # build token -> trajectory path
    token_to_traj = {}

    for traj_path in cache_dir.rglob("trajectory.gz"):
        token = traj_path.parent.name
        token_to_traj[token] = traj_path

    print("Cached tokens:", len(token_to_traj))

    samples = []

    with open(args.labels_path) as f:

        for line in f:
            rec = json.loads(line)

            tok = rec["token"]

            if tok in token_to_traj:
                samples.append(tok)

            if len(samples) >= args.num_samples:
                break

    print("\nChecking tokens:\n")

    for tok in samples:

        traj_path = token_to_traj[tok]

        traj = load_cached_traj(traj_path)

        cached_ep = traj[-1]

        print("token:", tok)
        print("cached endpoint:", cached_ep)

        # NOTE:
        # here you must retrieve the scenario object using your scenario builder
        # scenario = scenario_builder.get_scenario_from_token(tok)

        # classifier_ep = compute_endpoint_from_scenario(scenario)

        # print("classifier endpoint:", classifier_ep)

        # diff, diff_flip = compare_frames(cached_ep, classifier_ep)

        # print("distance:", diff)
        # print("distance if flip-y:", diff_flip)

        print()

if __name__ == "__main__":
    main()
