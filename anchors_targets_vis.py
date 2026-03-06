#!/usr/bin/env python3

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
    "others",
]


COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "black",
]


# --------------------------------------------------------
# Load scene labels
# --------------------------------------------------------

def load_scene_labels(labels_path):

    labels = {}
    class_tokens = defaultdict(list)

    with open(labels_path, "r") as f:
        for line in f:
            data = json.loads(line)
            token = data["token"]
            cls = data["emoe_class_id"]

            labels[token] = cls
            class_tokens[cls].append(token)

    return labels, class_tokens


# --------------------------------------------------------
# Load anchors
# --------------------------------------------------------

def load_anchors(anchor_path):

    anchors = np.load(anchor_path)  # [7, Ka, 2]
    return anchors


# --------------------------------------------------------
# Load scenarios
# --------------------------------------------------------

def build_scenarios(split):

    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / split

    worker = SingleMachineParallelExecutor(use_process_pool=False, num_workers=8)

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        log_names=None,
        map_names=None,
        num_scenarios=None,
        limit_total_scenarios=None,
    )

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=8,
    )

    scenarios = builder.get_scenarios(scenario_filter, worker)

    return {s.token: s for s in scenarios}


# --------------------------------------------------------
# Extract trajectory
# --------------------------------------------------------

def extract_trajectory(scenario):

    xs = []
    ys = []

    for i in range(scenario.get_number_of_iterations()):

        ego = scenario.get_ego_state_at_iteration(i)

        xs.append(ego.rear_axle.x)
        ys.append(ego.rear_axle.y)

    xs = np.array(xs)
    ys = np.array(ys)

    x0 = xs[0]
    y0 = ys[0]

    xs = xs - x0
    ys = ys - y0

    return xs, ys


# --------------------------------------------------------
# Collect trajectories per class
# --------------------------------------------------------

def collect_trajectories(class_tokens, scenarios, max_per_class=2000):

    trajectories = defaultdict(list)
    endpoints = defaultdict(list)

    for cls in class_tokens:

        tokens = class_tokens[cls][:max_per_class]

        for token in tqdm(tokens, desc=f"class {cls}"):

            if token not in scenarios:
                continue

            xs, ys = extract_trajectory(scenarios[token])

            trajectories[cls].append((xs, ys))

            endpoints[cls].append([xs[-1], ys[-1]])

    return trajectories, endpoints


# --------------------------------------------------------
# Plot endpoints
# --------------------------------------------------------

def plot_endpoints(endpoints, anchors, out_dir):

    plt.figure(figsize=(8,8))

    for cls, pts in endpoints.items():

        pts = np.array(pts)

        plt.scatter(
            pts[:,0],
            pts[:,1],
            s=5,
            alpha=0.5,
            color=COLORS[cls],
            label=EMOE_SCENE_TYPES[cls],
        )

        a = anchors[cls]

        plt.scatter(
            a[:,0],
            a[:,1],
            s=200,
            marker="*",
            color="black",
        )

    plt.legend()
    plt.title("Trajectory Endpoints + Anchors")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(out_dir / "endpoints_anchors.png", dpi=300)
    plt.close()


# --------------------------------------------------------
# Plot full trajectories
# --------------------------------------------------------

def plot_trajectories(trajectories, anchors, out_dir):

    plt.figure(figsize=(8,8))

    for cls, trajs in trajectories.items():

        for xs, ys in trajs:

            plt.plot(
                xs,
                ys,
                color=COLORS[cls],
                alpha=0.05,
            )

        a = anchors[cls]

        plt.scatter(
            a[:,0],
            a[:,1],
            s=200,
            marker="*",
            color="black",
        )

    plt.title("Full Trajectories + Anchors")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(out_dir / "trajectories_anchors.png", dpi=300)
    plt.close()


# --------------------------------------------------------
# Density heatmap
# --------------------------------------------------------

def plot_density(endpoints, out_dir):

    for cls, pts in endpoints.items():

        pts = np.array(pts)

        plt.figure(figsize=(6,6))

        sns.kdeplot(
            x=pts[:,0],
            y=pts[:,1],
            fill=True,
            cmap="viridis",
        )

        plt.title(EMOE_SCENE_TYPES[cls])
        plt.xlabel("x")
        plt.ylabel("y")

        plt.savefig(out_dir / f"density_class_{cls}.png", dpi=300)
        plt.close()


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():

    labels_path = Path("scene_labels.jsonl")
    anchors_path = Path("scene_anchors.npy")
    split = "mini"

    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    print("Loading labels...")
    labels, class_tokens = load_scene_labels(labels_path)

    print("Loading anchors...")
    anchors = load_anchors(anchors_path)

    print("Loading scenarios...")
    scenarios = build_scenarios(split)

    print("Collecting trajectories...")
    trajectories, endpoints = collect_trajectories(class_tokens, scenarios)

    print("Plotting endpoints...")
    plot_endpoints(endpoints, anchors, out_dir)

    print("Plotting trajectories...")
    plot_trajectories(trajectories, anchors, out_dir)

    print("Plotting density...")
    plot_density(endpoints, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
