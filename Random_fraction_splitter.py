from typing import List
import random

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class RandomFractionSplitter(AbstractSplitter):
    def __init__(self, train_fraction: float = 0.9, val_fraction: float = 0.1, seed: int = 0):
        assert train_fraction + val_fraction <= 1.0, "Fractions must sum to <= 1"
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.seed = seed

    def _split(self, scenarios: List[AbstractScenario]) -> (List[AbstractScenario], List[AbstractScenario]):
        random.seed(self.seed)
        random.shuffle(scenarios)
        n = len(scenarios)
        n_train = int(n * self.train_fraction)
        return scenarios[:n_train], scenarios[n_train:]

    def get_train_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool):
        train, _ = self._split(scenarios)
        return train

    def get_val_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool):
        _, val = self._split(scenarios)
        return val

    def get_test_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool):
        return []
