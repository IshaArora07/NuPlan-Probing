# src/custom_training/batched_caching.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.training.experiments.caching import cache_data as nuplan_cache_data
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _load_token_file(path: str) -> List[str]:
    """
    Load scenario tokens from a flat text file (one token per line).
    Skips blank lines and strips whitespace.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Token file not found: {path}")

    tokens = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    if not tokens:
        raise ValueError(f"Token file is empty: {path}")

    logger.info(f"Loaded {len(tokens):,} tokens from {path}")
    return tokens


def _make_chunks(tokens: List[str], size: int) -> List[List[str]]:
    """
    Split token list into chunks of at most `size`
    to stay safely under SQLite's 999-variable limit.
    """
    return [tokens[i : i + size] for i in range(0, len(tokens), size)]


# ─────────────────────────────────────────────────────────────
# Caching: drop-in replacement for nuPlan cache_data
# ─────────────────────────────────────────────────────────────
def cache_data_batched(cfg: DictConfig, worker) -> None:
    """
    Drop-in replacement for nuplan's cache_data.

    Reads scenario tokens from cfg.token_file, splits them into safe
    chunks of cfg.chunk_size, and calls nuplan cache_data once per chunk.

    If cfg.token_file is not set, falls back to standard cache_data.
    """
    token_file = OmegaConf.select(cfg, "token_file", default=None)
    chunk_size = int(OmegaConf.select(cfg, "chunk_size", default=900))

    if token_file is None:
        logger.info("No token_file set — using standard nuplan cache_data.")
        nuplan_cache_data(cfg=cfg, worker=worker)
        return

    all_tokens = _load_token_file(token_file)
    chunks = _make_chunks(all_tokens, chunk_size)

    logger.info(
        f"cache_data_batched: {len(all_tokens):,} tokens → "
        f"{len(chunks)} batches of ≤{chunk_size}"
    )

    for idx, chunk in enumerate(chunks):
        logger.info(
            f"Batch {idx + 1}/{len(chunks)}: caching {len(chunk)} tokens..."
        )

        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = chunk

        nuplan_cache_data(cfg=cfg, worker=worker)

        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = None

        logger.info(f"Batch {idx + 1}/{len(chunks)}: done.")

    logger.info("cache_data_batched: all batches complete.")


# ─────────────────────────────────────────────────────────────
# Training: batched scenario list builder
# ─────────────────────────────────────────────────────────────
def build_scenarios_batched(
    cfg: DictConfig,
    worker,
    model: TorchModuleWrapper,
) -> list:
    """
    Batched replacement for nuplan's build_scenarios.

    Reads tokens from cfg.token_file, queries DB in chunks of
    cfg.chunk_size, and returns one merged list of scenarios.

    Training itself remains a single continuous pass.
    """
    token_file = OmegaConf.select(cfg, "token_file", default=None)
    chunk_size = int(OmegaConf.select(cfg, "chunk_size", default=900))

    if token_file is None:
        logger.info("No token_file set — using standard nuplan build_scenarios.")
        return build_scenarios(cfg, worker, model)

    all_tokens = _load_token_file(token_file)
    chunks = _make_chunks(all_tokens, chunk_size)

    logger.info(
        f"build_scenarios_batched: {len(all_tokens):,} tokens → "
        f"{len(chunks)} batches of ≤{chunk_size}"
    )

    all_scenarios = []

    for idx, chunk in enumerate(chunks):
        logger.info(
            f"Batch {idx + 1}/{len(chunks)}: fetching {len(chunk)} tokens..."
        )

        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = chunk

        batch = build_scenarios(cfg, worker, model)
        all_scenarios.extend(batch)

        with open_dict(cfg):
            cfg.scenario_filter.scenario_tokens = None

        logger.info(
            f"Batch {idx + 1}/{len(chunks)}: got {len(batch)} scenarios "
            f"(running total: {len(all_scenarios):,})"
        )

    logger.info(
        f"build_scenarios_batched: done. "
        f"Total scenarios handed to DataModule: {len(all_scenarios):,}"
    )

    return all_scenarios
