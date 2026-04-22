#!/usr/bin/env python3
"""
load_utils.py
Shared utilities for EMoE simulation analysis.
Used by analyse_core.py, analyse_behaviour.py, analyse_improvement.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# CONFIG — edit these to match your local setup
# ─────────────────────────────────────────────

# Map run name → local folder path
RUN_PATHS = {
    "T2": os.path.expanduser("~/Desktop/Thesis/Training2"),
    "T3": os.path.expanduser("~/Desktop/Thesis/Training3"),
    "T4": os.path.expanduser("~/Desktop/Thesis/Training4"),
    # Add T6 etc. here when ready
}

# Primary run for single-run analyses
PRIMARY_RUN = "T3"

# Output directory for plots, CSVs, HTML
OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/Thesis/analysis_output"))

# Thresholds for failure detection
HARD_BRAKE_THRESHOLD = -3.0
HIGH_LAT_ACCEL = 2.5
HIGH_YAW_RATE = 0.5
PROGRESS_FAIL_THRESH = 0.5
TTC_FAIL_THRESH = 1.0

# ─────────────────────────────────────────────
# KNOWN METRIC FILES inside metrics/
# ─────────────────────────────────────────────

METRIC_FILES = [
    "corners_in_drivable_area",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_comfortable",
    "ego_is_making_progress",
    "ego_jerk",
    "ego_lane_change",
    "ego_lat_acceleration",
    "ego_lon_acceleration",
    "ego_lon_jerk",
    "ego_progress_along_expert_route",
    "ego_yaw_acceleration",
    "ego_yaw_rate",
    "no_ego_at_fault_collisions",
    "speed_limit_compliance",
    "time_to_collision_within_bound",
]

# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────

def get_run_path(run_name: str) -> Path:
    assert run_name in RUN_PATHS, f"Unknown run: {run_name}. Add it to RUN_PATHS"
    return Path(RUN_PATHS[run_name])


def get_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_metric_path(run_name: str, metric_name: str) -> Path:
    return get_run_path(run_name) / "metrics" / f"{metric_name}.parquet"


def get_aggregator_path(run_name: str) -> Path:
    agg_dir = get_run_path(run_name) / "aggregator_metric"
    candidates = list(agg_dir.glob("*.parquet"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No parquet found in {agg_dir}")
    return candidates[0]


def get_runner_report_path(run_name: str) -> Path:
    return get_run_path(run_name) / "runner_report.parquet"


def get_summary_dir(run_name: str) -> Path:
    return get_run_path(run_name) / "summary"

# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

def load_runner_report(run_name: str) -> pd.DataFrame:
    path = get_runner_report_path(run_name)
    df = pd.read_parquet(path)
    df = _detect_and_rename_scenario_columns(df)
    print(f"[{run_name}] runner_report: {len(df)} rows")
    return df


def load_aggregator(run_name: str) -> pd.DataFrame:
    path = get_aggregator_path(run_name)
    df = pd.read_parquet(path)
    print(f"[{run_name}] aggregator_metric: {len(df)} rows")
    return df


def load_metric(run_name: str, metric_name: str) -> pd.DataFrame | None:
    path = get_metric_path(run_name, metric_name)
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    df = pd.read_parquet(path)
    df = _detect_and_rename_scenario_columns(df)
    return df


def load_all_metrics(run_name: str) -> pd.DataFrame:
    """
    Load all per-scenario metric parquets and merge into one dataframe.
    """
    merged = None
    join_col = None

    for metric_name in METRIC_FILES:
        df = load_metric(run_name, metric_name)
        if df is None:
            continue

        if join_col is None:
            join_col = _find_join_key(df)

        val_col = _find_value_column(df, exclude=[join_col])
        if val_col is None:
            continue

        sub = df[[join_col, val_col]].rename(columns={val_col: metric_name})

        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on=join_col, how="outer")

    if merged is None:
        raise RuntimeError(f"No metrics loaded for {run_name}")

    print(f"[{run_name}] merged metrics: {len(merged)} scenarios")
    return merged

# ─────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────

SCENARIO_TYPE_CANDIDATES = [
    "scenario_type", "type", "scenario_name", "scene_type"
]

SCENARIO_TOKEN_CANDIDATES = [
    "token", "scenario_token", "scenario_id", "id"
]


def _detect_and_rename_scenario_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.lower() for c in df.columns]
    rename_map = {}

    for c in SCENARIO_TYPE_CANDIDATES:
        if c in cols_lower:
            rename_map[df.columns[cols_lower.index(c)]] = "scenario_type"
            break

    for c in SCENARIO_TOKEN_CANDIDATES:
        if c in cols_lower:
            rename_map[df.columns[cols_lower.index(c)]] = "token"
            break

    return df.rename(columns=rename_map) if rename_map else df


def _find_join_key(df: pd.DataFrame) -> str:
    for col in ["token", "scenario_token", "scenario_id", "id"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _find_value_column(df: pd.DataFrame, exclude: list) -> str | None:
    for col in df.columns:
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def get_scenario_type_col(df: pd.DataFrame) -> str | None:
    return "scenario_type" if "scenario_type" in df.columns else None

# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────

PALETTE = {
    "T2": "#4C9BE8",
    "T3": "#F28C38",
    "T4": "#6BCB77",
}

def set_plot_style():
    plt.style.use("default")


def save_fig(fig, name: str, subdir: str = ""):
    out = get_output_dir()
    if subdir:
        out = out / subdir
    out.mkdir(parents=True, exist_ok=True)

    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


def bar_chart(ax, labels, values, color="#F28C38", title="", ylabel="", ylim=None):
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
