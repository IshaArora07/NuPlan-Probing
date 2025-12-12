#!/usr/bin/env python3
"""
Analyze EMoE classification outputs from scene_labels.jsonl and save plots + CSVs.

Outputs in --out_dir:
  - 01_class_distribution.png
  - 02_stage_distribution.png
  - 03_class_stage_heatmap.png
  - 04_heading_hist_by_class.png
  - 05_others_breakdown_stage.png
  - 06_others_breakdown_intersection_map.png
  - 07_others_breakdown_connector_best_type.png
  - 08_tag_vs_map_disagreement.png
  - 09_anchor_scatter.png (optional, if --anchors_npy provided)

Plus CSVs:
  - class_counts.csv
  - stage_counts.csv
  - class_by_stage_counts.csv
  - tag_map_disagreement.csv

Usage:
  python analyze_scene_labels.py \
    --scene_labels /path/to/scene_labels.jsonl \
    --out_dir ./analysis_out \
    --anchors_npy /path/to/scene_anchors.npy
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",      # 0
    "straight_at_intersection",       # 1
    "right_turn_at_intersection",     # 2
    "straight_non_intersection",      # 3
    "roundabout",                     # 4
    "u_turn",                         # 5
    "others",                         # 6
]


# -----------------------------
# Loading
# -----------------------------
def load_scene_labels_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            flat: Dict[str, Any] = {
                "token": obj.get("token", ""),
                "class_id": obj.get("emoe_class_id", None),
                "class_name": obj.get("emoe_class_name", ""),
                "stage": obj.get("stage", ""),
                "scenario_type": obj.get("scenario_type", ""),
                "travel_distance_m": obj.get("travel_distance_m", None),
            }

            debug = obj.get("debug", {}) or {}
            # flatten debug
            for k, v in debug.items():
                flat[f"debug_{k}"] = v

            # convenience: normalize some expected debug keys
            # (some may be missing depending on your script version)
            rows.append(flat)

    df = pd.DataFrame(rows)

    # Fix dtypes where possible
    for col in ["class_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing class_name if needed
    if "class_name" in df.columns:
        missing = df["class_name"].isna() | (df["class_name"] == "")
        if missing.any() and "class_id" in df.columns:
            def _name(cid):
                try:
                    cid = int(cid)
                    if 0 <= cid < len(EMOE_SCENE_TYPES):
                        return EMOE_SCENE_TYPES[cid]
                except Exception:
                    pass
                return "unknown"
            df.loc[missing, "class_name"] = df.loc[missing, "class_id"].apply(_name)

    return df


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------------
# Plot helpers
# -----------------------------
def bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path, rotate: int = 30) -> None:
    plt.figure(figsize=(10, 4))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save_fig(out_path)


def heatmap_counts(df_counts: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    Matplotlib-only heatmap (no seaborn dependency).
    """
    data = df_counts.values.astype(float)

    plt.figure(figsize=(max(10, 0.55 * df_counts.shape[1]), max(4.5, 0.45 * df_counts.shape[0])))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title(title)

    plt.yticks(range(df_counts.shape[0]), df_counts.index.tolist())
    plt.xticks(range(df_counts.shape[1]), df_counts.columns.tolist(), rotation=45, ha="right")

    # annotate (only if not huge)
    if df_counts.shape[0] * df_counts.shape[1] <= 350:
        for i in range(df_counts.shape[0]):
            for j in range(df_counts.shape[1]):
                plt.text(j, i, f"{int(df_counts.iloc[i, j])}", ha="center", va="center", fontsize=8)

    save_fig(out_path)


# -----------------------------
# Analyses
# -----------------------------
def run_all(df: pd.DataFrame, out_dir: Path, anchors_npy: Optional[Path]) -> None:
    # 1) Class distribution
    class_counts = df["class_name"].value_counts()
    class_counts.to_csv(out_dir / "class_counts.csv", header=["count"])
    bar_plot(
        class_counts,
        title="EMoE class distribution",
        xlabel="class",
        ylabel="count",
        out_path=out_dir / "01_class_distribution.png",
        rotate=25,
    )

    # 2) Stage distribution
    stage_counts = df["stage"].value_counts()
    stage_counts.to_csv(out_dir / "stage_counts.csv", header=["count"])
    bar_plot(
        stage_counts,
        title="Which stage classified how many scenarios",
        xlabel="stage",
        ylabel="count",
        out_path=out_dir / "02_stage_distribution.png",
        rotate=45,
    )

    # 3) Class × Stage matrix
    pivot = df.pivot_table(
        index="class_name",
        columns="stage",
        values="token",
        aggfunc="count",
        fill_value=0,
    ).astype(int)
    pivot.to_csv(out_dir / "class_by_stage_counts.csv")
    heatmap_counts(
        pivot,
        title="Class × Stage classification matrix (counts)",
        out_path=out_dir / "03_class_stage_heatmap.png",
    )

    # 4) Heading distributions per class (if available)
    heading_col = "debug_abs_delta_heading_deg"
    if heading_col in df.columns:
        plt.figure(figsize=(10, 6))
        for cname in sorted(df["class_name"].unique()):
            sub = df[df["class_name"] == cname]
            vals = pd.to_numeric(sub[heading_col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=40, alpha=0.35, label=cname)

        plt.axvline(35, linestyle="--", label="turn threshold (35°)")
        plt.title("Net heading change |Δheading| distribution by class")
        plt.xlabel("|Δ heading| (deg)")
        plt.ylabel("count")
        plt.legend(loc="upper right", fontsize=8)
        save_fig(out_dir / "04_heading_hist_by_class.png")
    else:
        # still create a note file so you know why it didn't plot
        (out_dir / "04_heading_hist_by_class.SKIPPED.txt").write_text(
            f"Missing column '{heading_col}' in scene_labels.jsonl debug.\n"
        )

    # 5) OTHERS breakdown
    others = df[df["class_name"] == "others"]

    if len(others) > 0:
        others_stage = others["stage"].value_counts()
        bar_plot(
            others_stage,
            title="OTHERS: breakdown by stage",
            xlabel="stage",
            ylabel="count",
            out_path=out_dir / "05_others_breakdown_stage.png",
            rotate=45,
        )

        inter_map_col = "debug_has_intersection_map"
        if inter_map_col in others.columns:
            inter_map_counts = others[inter_map_col].value_counts(dropna=False)
            bar_plot(
                inter_map_counts,
                title="OTHERS: has_intersection_map distribution",
                xlabel="has_intersection_map",
                ylabel="count",
                out_path=out_dir / "06_others_breakdown_intersection_map.png",
                rotate=0,
            )
        else:
            (out_dir / "06_others_breakdown_intersection_map.SKIPPED.txt").write_text(
                f"Missing column '{inter_map_col}' in debug.\n"
            )

        conn_best_col = "debug_connector_best_type"
        if conn_best_col in others.columns:
            conn_best_counts = others[conn_best_col].value_counts(dropna=False)
            bar_plot(
                conn_best_counts,
                title="OTHERS: connector best_type distribution",
                xlabel="connector_best_type",
                ylabel="count",
                out_path=out_dir / "07_others_breakdown_connector_best_type.png",
                rotate=25,
            )
        else:
            (out_dir / "07_others_breakdown_connector_best_type.SKIPPED.txt").write_text(
                f"Missing column '{conn_best_col}' in debug.\n"
            )
    else:
        (out_dir / "05_06_07_OTHERS.SKIPPED.txt").write_text("No rows with class_name == 'others'.\n")

    # 6) Tag vs map disagreement
    tag_col = "debug_has_intersection_tag"
    map_col = "debug_has_intersection_map"
    if tag_col in df.columns and map_col in df.columns:
        tag_vals = df[tag_col].fillna(False).astype(bool)
        map_vals = df[map_col].fillna(False).astype(bool)

        a = int(((tag_vals == True) & (map_vals == False)).sum())
        b = int(((tag_vals == False) & (map_vals == True)).sum())
        c = int(((tag_vals == True) & (map_vals == True)).sum())
        d = int(((tag_vals == False) & (map_vals == False)).sum())

        disagreement_df = pd.DataFrame(
            [
                ["tag=True, map=False", a],
                ["tag=False, map=True", b],
                ["tag=True, map=True", c],
                ["tag=False, map=False", d],
            ],
            columns=["case", "count"],
        )
        disagreement_df.to_csv(out_dir / "tag_map_disagreement.csv", index=False)

        bar_plot(
            disagreement_df.set_index("case")["count"],
            title="Tag vs map intersection agreement/disagreement",
            xlabel="case",
            ylabel="count",
            out_path=out_dir / "08_tag_vs_map_disagreement.png",
            rotate=25,
        )
    else:
        (out_dir / "08_tag_vs_map_disagreement.SKIPPED.txt").write_text(
            f"Missing '{tag_col}' and/or '{map_col}' in debug.\n"
        )

    # 7) Anchor scatter (optional)
    if anchors_npy is not None and anchors_npy.exists():
        anchors = np.load(str(anchors_npy))
        if anchors.ndim == 3 and anchors.shape[0] == 7 and anchors.shape[2] == 2:
            plt.figure(figsize=(9, 6))
            for c in range(7):
                pts = anchors[c]
                plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f"{c}:{EMOE_SCENE_TYPES[c]}")
            plt.axhline(0)
            plt.axvline(0)
            plt.axis("equal")
            plt.title("Scene anchors (endpoint cluster centers) per class")
            plt.xlabel("x_ego [m]")
            plt.ylabel("y_ego [m]")
            plt.legend(fontsize=7, loc="best")
            save_fig(out_dir / "09_anchor_scatter.png")
        else:
            (out_dir / "09_anchor_scatter.SKIPPED.txt").write_text(
                f"Anchors array has unexpected shape: {anchors.shape}. Expected [7, Ka, 2].\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_labels", type=str, required=True, help="Path to scene_labels.jsonl")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for plots + CSVs")
    parser.add_argument("--anchors_npy", type=str, default="", help="Optional path to scene_anchors.npy")
    args = parser.parse_args()

    scene_labels = Path(args.scene_labels).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    anchors_npy = Path(args.anchors_npy).expanduser().resolve() if args.anchors_npy else None

    ensure_out_dir(out_dir)

    print(f"[INFO] Loading: {scene_labels}")
    df = load_scene_labels_jsonl(scene_labels)
    print(f"[INFO] Loaded {len(df)} rows")

    # Quick metadata dump
    (out_dir / "README.txt").write_text(
        f"Input: {scene_labels}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {list(df.columns)}\n"
        f"Anchors: {anchors_npy if anchors_npy else 'None'}\n"
    )

    run_all(df, out_dir, anchors_npy)

    print(f"[DONE] Saved analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
