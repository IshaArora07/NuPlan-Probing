# src/custom_training/batched_caching.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.training.experiments.caching import cache_data as nuplan_cache_data
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


def _load_token_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token file not found: {path}")
    tokens = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"Token file is empty: {path}")
    logger.info(f"Loaded {len(tokens):,} tokens from {path}")
    return tokens


def _make_chunks(tokens: List[str], size: int) -> List[List[str]]:
    return [tokens[i : i + size] for i in range(0, len(tokens), size)]


def cache_data_batched(cfg: DictConfig, worker) -> None:
    """
    Drop-in replacement for nuplan's cache_data.

    Reads scenario tokens from a file specified in cfg.token_file,
    splits into safe chunks, and calls nuplan's cache_data once per chunk.

    If cfg.token_file is not set, falls back to nuplan's cache_data unchanged.

    In your run_training.py, replace:
        cache_data(cfg=cfg, worker=worker)
    with:
        cache_data_batched(cfg=cfg, worker=worker)

    In your caching shell script, pass:
        +token_file=/path/to/tokens.txt
        +chunk_size=900
    """

    # Read our custom top-level params (not inside scenario_filter)
    token_file = OmegaConf.select(cfg, "token_file", default=None)
    chunk_size  = int(OmegaConf.select(cfg, "chunk_size", default=900))

    if token_file is None:
        # No token file — pass straight through, zero changes
        logger.info("No token_file set — using standard nuplan cache_data.")
        nuplan_cache_data(cfg=cfg, worker=worker)
        return

    # Load and chunk tokens
    all_tokens = _load_token_file(token_file)
    chunks = _make_chunks(all_tokens, chunk_size)
    logger.info(
        f"cache_data_batched: {len(all_tokens):,} tokens → "
        f"{len(chunks)} batches of ≤{chunk_size}"
    )

    for idx, chunk in enumerate(chunks):
        logger.info(f"Batch {idx + 1}/{len(chunks)}: processing {len(chunk)} tokens...")

        # Patch ONLY scenario_filter.scenario_tokens for this batch.
        # We keep token_file/chunk_size at the top level (not in scenario_filter)
        # so build_scenario_filter never sees them and ScenarioFilter.__init__
        # never receives unknown kwargs.
        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = chunk

        nuplan_cache_data(cfg=cfg, worker=worker)

        # Reset for next batch
        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = None

        logger.info(f"Batch {idx + 1}/{len(chunks)}: done.")

    logger.info("cache_data_batched: all batches complete.")
