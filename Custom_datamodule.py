import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data import WeightedRandomSampler

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_loader.distributed_sampler_wrapper import (
    DistributedSamplerWrapper,
)
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    move_features_type_to_device,
)
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError(
    'Data module has not been setup, call "setup()"'
)


def create_dataset(
    samples: List[AbstractScenario],
    feature_preprocessor: FeaturePreprocessor,
    dataset_fraction: float,
    dataset_name: str,
    augmentors: Optional[List[AbstractAugmentor]] = None,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    """
    num_keep = max(1, int(len(samples) * dataset_fraction))
    selected_scenarios = random.sample(samples, num_keep)

    logger.info(
        f"Number of samples in {dataset_name} set: {len(selected_scenarios)}"
    )

    return ScenarioDataset(
        scenarios=selected_scenarios,
        feature_preprocessor=feature_preprocessor,
        augmentors=augmentors,
    )


def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,
    scenario_sampling_weights: Dict[str, float],
    replacement: bool = True,
) -> DistributedSamplerWrapper:
    """
    Standard nuPlan scenario-type weighted sampler.
    """
    scenarios = scenario_dataset._scenarios

    if not replacement:
        assert all(
            w > 0 for w in scenario_sampling_weights.values()
        ), "All scenario sampling weights must be positive"

    default_weight = 1.0

    sample_weights = [
        scenario_sampling_weights.get(
            scenario.scenario_type,
            default_weight,
        )
        for scenario in scenarios
    ]

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(scenarios),
        replacement=replacement,
    )

    return DistributedSamplerWrapper(weighted_sampler)


def emoe_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,
    scene_labels_path: str,
    num_classes: int = 6,
    replacement: bool = True,
) -> DistributedSamplerWrapper:
    """
    EMoE inverse-frequency weighted sampler using scene_labels.jsonl.
    """
    token_to_class: Dict[str, int] = {}
    counts = np.zeros(num_classes, dtype=np.float64)

    with open(scene_labels_path, "r") as f:
        for line in f:
            record = json.loads(line)
            token = record["token"]
            cls = int(record.get("emoe_class_id", -1))

            token_to_class[token] = cls

            if 0 <= cls < num_classes:
                counts[cls] += 1

    logger.info(
        f"[EMoE Sampler] Class counts: {counts.astype(int).tolist()}"
    )

    class_weights = 1.0 / np.maximum(counts, 1.0)
    mean_weight = float(class_weights.mean())

    scenarios = scenario_dataset._scenarios
    labels = np.array(
        [token_to_class.get(s.token, -1) for s in scenarios],
        dtype=np.int64,
    )

    sample_weights = np.where(
        labels >= 0,
        class_weights[np.clip(labels, 0, num_classes - 1)],
        mean_weight,
    )

    n_unknown = int((labels < 0).sum())
    if n_unknown > 0:
        logger.warning(
            f"[EMoE Sampler] {n_unknown} unknown scenario tokens"
        )

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(scenarios),
        replacement=replacement,
    )

    return DistributedSamplerWrapper(weighted_sampler)


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule for PLUTO + EMoE.
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        super().__init__()

        assert train_fraction > 0.0
        assert val_fraction > 0.0
        assert test_fraction >= 0.0

        self._train_set = None
        self._val_set = None
        self._test_set = None

        self._feature_preprocessor = feature_preprocessor
        self._splitter = splitter
        self._all_samples = all_scenarios
        self._worker = worker
        self._augmentors = augmentors

        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction
        self._dataloader_params = dataloader_params
        self._scenario_type_sampling_weights = scenario_type_sampling_weights

        assert len(self._all_samples) > 0, "No samples passed"

        # FIXED: correct env var name
        self._scene_labels_path = os.environ.get(
            "EMOE_SCENE_LABELS_PATH",
            None,
        )

        if self._scene_labels_path:
            logger.info(
                f"[EMoE Sampler] scene_labels_path={self._scene_labels_path}"
            )
        else:
            logger.warning(
                "[EMoE Sampler] EMOE_SCENE_LABELS_PATH not set"
            )

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        return self._feature_preprocessor

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            train_samples = self._splitter.get_train_samples(
                self._all_samples, self._worker
            )
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )

            self._train_set = create_dataset(
                train_samples,
                self._feature_preprocessor,
                self._train_fraction,
                "train",
                self._augmentors,
            )

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )

        elif stage == "validate":
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )

        elif stage == "test":
            test_samples = self._splitter.get_test_samples(
                self._all_samples, self._worker
            )

            self._test_set = create_dataset(
                test_samples,
                self._feature_preprocessor,
                self._test_fraction,
                "test",
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self._train_set is None:
            raise DataModuleNotSetupError

        weighted_sampler = None

        if self._scenario_type_sampling_weights.enable:
            if self._scene_labels_path:
                logger.info("[EMoE Sampler] Using EMoE weighted sampler")
                weighted_sampler = emoe_weighted_sampler_init(
                    self._train_set,
                    self._scene_labels_path,
                )
            else:
                logger.info("[EMoE Sampler] Falling back to nuPlan weights")
                weighted_sampler = distributed_weighted_sampler_init(
                    self._train_set,
                    self._scenario_type_sampling_weights.scenario_type_weights,
                )

        return torch.utils.data.DataLoader(
            dataset=self._train_set,
            shuffle=weighted_sampler is None,
            sampler=weighted_sampler,
            collate_fn=FeatureCollate(),
            **self._dataloader_params,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self._val_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._val_set,
            collate_fn=FeatureCollate(),
            **self._dataloader_params,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self._test_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._test_set,
            collate_fn=FeatureCollate(),
            **self._dataloader_params,
        )

    def transfer_batch_to_device(
        self,
        batch: Tuple[FeaturesType, ...],
        device: torch.device,
        dataloader_idx: int,
    ) -> Tuple[FeaturesType, ...]:
        return (
            move_features_type_to_device(batch[0], device),
            move_features_type_to_device(batch[1], device),
            batch[2],
        )
