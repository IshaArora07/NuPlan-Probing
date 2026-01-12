#!/usr/bin/env python3
"""
Clean a scenarios.yaml by removing scenario_tokens that lead to
route/goal errors (e.g. 'NoneType object has no attribute id').

Usage:
    python clean_scenarios_yaml.py \
        --input_yaml config/training/mytokens.yaml \
        --output_yaml config/training/mytokens_clean.yaml \
        --split trainval \
        --num_workers 8

Requires:
    - NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT set in the environment
    - nuplan-devkit installed / on PYTHONPATH
"""

import argparse
import os
from pathlib import Path

import yaml

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import SingleMachineParallelExecutor


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml", type=str, required=True)
    parser.add_argument("--output_yaml", type=str, required=True)
    parser.add_argument("--split", type=str, default="trainval",
                        help="nuPlan split: trainval, mini, val, test, ...")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    input_yaml = Path(args.input_yaml).expanduser().resolve()
    output_yaml = Path(args.output_yaml).expanduser().resolve()
    bad_tokens_txt = output_yaml.with_suffix(".dropped_tokens.txt")

    cfg = load_yaml(input_yaml)

    # ------------------------------------------------------------------
    # 1) Read token list from YAML
    # ------------------------------------------------------------------
    # Support either "scenario_tokens" or "emoe_tokens" field name:
    tokens_key = "scenario_tokens"
    if tokens_key not in cfg:
        if "emoe_tokens" in cfg:
            tokens_key = "emoe_tokens"
        else:
            raise KeyError("No 'scenario_tokens' or 'emoe_tokens' field found in YAML.")

    original_tokens = [str(t) for t in (cfg.get(tokens_key) or [])]
    if not original_tokens:
        raise ValueError("Token list in YAML is empty.")

    print(f"[INFO] Loaded {len(original_tokens)} tokens from {input_yaml}")

    # ------------------------------------------------------------------
    # 2) Build ScenarioFilter with remove_invalid_goals=True
    # ------------------------------------------------------------------
    # Carry over other filter fields if present; otherwise use defaults.
    sf_kwargs = dict(
        scenario_types=cfg.get("scenario_types", None),
        scenario_tokens=original_tokens,
        log_names=cfg.get("log_names", None),
        map_names=cfg.get("map_names", None),
        num_scenarios_per_type=cfg.get("num_scenarios_per_type", None),
        limit_total_scenarios=cfg.get("limit_total_scenarios", None),
        timestamp_threshold_s=cfg.get("timestamp_threshold_s", 15),
        ego_displacement_minimum_m=cfg.get("ego_displacement_minimum_m", None),
        ego_start_speed_threshold=cfg.get("ego_start_speed_threshold", None),
        ego_stop_speed_threshold=cfg.get("ego_stop_speed_threshold", None),
        speed_noise_tolerance=cfg.get("speed_noise_tolerance", None),
        expand_scenarios=cfg.get("expand_scenarios", None),
        remove_invalid_goals=True,   # <-- crucial to drop the bad ones
        shuffle=cfg.get("shuffle", False),
    )

    scenario_filter = ScenarioFilter(**sf_kwargs)

    # ------------------------------------------------------------------
    # 3) Build ScenarioBuilder (nuplan DB) and get surviving scenarios
    # ------------------------------------------------------------------
    data_root = os.environ["NUPLAN_DATA_ROOT"]
    map_root = os.environ["NUPLAN_MAPS_ROOT"]

    db_root = Path(data_root) / "nuplan-v1.1" / "splits" / args.split
    if not db_root.exists():
        raise FileNotFoundError(f"DB split directory not found: {db_root}")

    builder = NuPlanScenarioBuilder(
        data_root=str(db_root),
        map_root=str(map_root),
        sensor_root=None,
        db_files=None,              # use all DB files in the split
        map_version="nuplan-maps-v1.0",
        include_cameras=False,
        max_workers=args.num_workers,
    )

    worker = SingleMachineParallelExecutor(
        use_process_pool=False,
        num_workers=args.num_workers,
    )

    print("[INFO] Building scenarios (this may take a while the first time)...")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"[INFO] ScenarioBuilder returned {len(scenarios)} scenarios after filtering.")

    good_tokens = [str(s.token) for s in scenarios]
    good_set = set(good_tokens)
    bad_tokens = [t for t in original_tokens if t not in good_set]

    print(f"[INFO] Good tokens : {len(good_tokens)}")
    print(f"[INFO] Dropped tokens (likely invalid goals / route issues): {len(bad_tokens)}")

    # ------------------------------------------------------------------
    # 4) Write cleaned YAML and list of dropped tokens
    # ------------------------------------------------------------------
    cfg_clean = dict(cfg)  # shallow copy
    cfg_clean[tokens_key] = good_tokens

    save_yaml(cfg_clean, output_yaml)
    print(f"[INFO] Wrote cleaned scenarios YAML to: {output_yaml}")

    if bad_tokens:
        with bad_tokens_txt.open("w") as f:
            for t in bad_tokens:
                f.write(f"{t}\n")
        print(f"[INFO] Wrote dropped tokens to: {bad_tokens_txt}")


if __name__ == "__main__":
    main()
