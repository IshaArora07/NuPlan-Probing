#!/usr/bin/env python3
"""
EMoE Anchor Diagnostic Visualizer

Produces three figures from scene_labels.jsonl + scene_anchors.npy
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",
    "straight_at_intersection",
    "right_turn_at_intersection",
    "straight_non_intersection",
    "roundabout",
    "u_turn",
]

SHORT_NAMES = [
    "Left turn\n(intersection)",
    "Straight\n(intersection)",
    "Right turn\n(intersection)",
    "Straight\n(non-intersection)",
    "Roundabout",
    "U-turn",
]

CLASS_COLORS = [
    "#E63946",
    "#2196F3",
    "#FF9800",
    "#4CAF50",
    "#9C27B0",
    "#00BCD4",
]

ANCHOR_COLOR = "#1A1A2E"
ANCHOR_MARKER = "*"
ANCHOR_SIZE = 160


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_labels(labels_path: Path, min_dist_m: float):
    endpoints = defaultdict(list)

    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue

            dist = rec.get("travel_distance_m", 0.0) or 0.0
            if dist < min_dist_m:
                continue

            ep = rec.get("endpoint_xy")
            if ep is None:
                continue

            cid = int(rec.get("emoe_class_id", 6))
            endpoints[cid].append(np.array(ep, dtype=np.float32))

    return endpoints


def load_anchors(anchors_path: Path):
    anchors = np.load(anchors_path)
    print(f"Anchors shape: {anchors.shape} dtype: {anchors.dtype}")
    return anchors


# ─────────────────────────────────────────────
# FIGURE 1 — SCATTER
# ─────────────────────────────────────────────

def plot_scatter(endpoints, anchors, max_pts, out_path):
    n_classes = anchors.shape[0]

    ncols = 3
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 6.5 * nrows),
        constrained_layout=True
    )

    axes = np.array(axes).flatten()
    rng = np.random.default_rng(seed=42)

    for c in range(n_classes):

        ax = axes[c]
        col = CLASS_COLORS[c % len(CLASS_COLORS)]
        name = SHORT_NAMES[c] if c < len(SHORT_NAMES) else f"class_{c}"

        pts = endpoints.get(c, [])
        n_total = len(pts)

        if pts:
            idx = rng.choice(len(pts), size=min(max_pts, len(pts)), replace=False)
            pts_plot = np.stack([pts[i] for i in idx])
        else:
            pts_plot = np.empty((0, 2), dtype=np.float32)

        if pts_plot.shape[0] > 0:
            ax.scatter(
                pts_plot[:, 0],
                pts_plot[:, 1],
                s=14,
                alpha=0.35,
                color=col,
                linewidths=0,
                zorder=2,
                label=f"GT endpoints (n={n_total})"
            )

        anc = anchors[c]

        ax.scatter(
            anc[:, 0],
            anc[:, 1],
            s=ANCHOR_SIZE,
            color=ANCHOR_COLOR,
            marker=ANCHOR_MARKER,
            zorder=5,
            linewidths=0.4,
            edgecolors="white",
            label=f"Anchors (Ka={anc.shape[0]})"
        )

        ax.scatter(0, 0, s=80, color="black", marker="D", zorder=6)

        ax.annotate(
            "",
            xy=(3, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

        ax.set_title(name, fontsize=13, fontweight="bold", color=col)
        ax.set_xlabel("x (ego-forward, m)")
        ax.set_ylabel("y (ego-left, m)")

        ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)

        if pts_plot.shape[0] > 0:
            all_x = np.concatenate([pts_plot[:, 0], anc[:, 0]])
            all_y = np.concatenate([pts_plot[:, 1], anc[:, 1]])

            margin = 5.0
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        ax.legend(fontsize=8, loc="upper left")

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "EMoE Planner — GT Endpoints vs KMeans Anchors",
        fontsize=14,
        fontweight="bold"
    )

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────
# FIGURE 2 — COVERAGE
# ─────────────────────────────────────────────

def plot_coverage(endpoints, anchors, out_path):

    n_classes = anchors.shape[0]
    ncols = 3
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )

    axes = np.array(axes).flatten()

    for c in range(n_classes):

        ax = axes[c]
        col = CLASS_COLORS[c % len(CLASS_COLORS)]
        name = SHORT_NAMES[c] if c < len(SHORT_NAMES) else f"class_{c}"

        pts = endpoints.get(c, [])

        if not pts:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(name)
            continue

        pts_arr = np.stack(pts)
        anc = anchors[c]

        diffs = pts_arr[:, None, :] - anc[None, :, :]
        min_dists = np.linalg.norm(diffs, axis=-1).min(axis=1)

        med = float(np.median(min_dists))
        p90 = float(np.percentile(min_dists, 90))

        ax.hist(min_dists, bins=40, color=col, alpha=0.75)

        ax.axvline(med, color="black", lw=1.8, label=f"median={med:.1f}")
        ax.axvline(p90, color="black", lw=1.4, ls="--", label=f"p90={p90:.1f}")

        ax.set_title(name, fontweight="bold", color=col)
        ax.set_xlabel("Distance to nearest anchor (m)")
        ax.set_ylabel("Count")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--anchors_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./emoe_viz")
    parser.add_argument("--max_pts", type=int, default=500)
    parser.add_argument("--min_dist_m", type=float, default=5.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")

    endpoints = load_labels(Path(args.labels_path), args.min_dist_m)
    anchors = load_anchors(Path(args.anchors_path))

    plot_scatter(
        endpoints,
        anchors,
        args.max_pts,
        out_dir / "fig1_scatter_endpoints_anchors.png"
    )

    plot_coverage(
        endpoints,
        anchors,
        out_dir / "fig2_anchor_coverage.png"
    )

    print(f"\nFigures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
