# src/splitters/token_list_splitter.py

from typing import List, Set, Tuple

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class TokenListSplitter(AbstractSplitter):
    """
    Deterministic splitter based on explicit scenario tokens.

    Rules:
      - token in val_tokens   -> val
      - token in train_tokens -> train
      - everything else       -> ignored (not used)
      - test is always empty
    """

    def __init__(self, train_tokens: List[str], val_tokens: List[str]):
        self._train_tokens: Set[str] = set(train_tokens)
        self._val_tokens: Set[str] = set(val_tokens)

        overlap = self._train_tokens & self._val_tokens
        assert len(overlap) == 0, f"Train/val token overlap detected: {len(overlap)} tokens"

    def _split(self, scenarios: List[AbstractScenario]) -> Tuple[List[AbstractScenario], List[AbstractScenario]]:
        train: List[AbstractScenario] = []
        val: List[AbstractScenario] = []

        for s in scenarios:
            # NuPlan scenarios expose .token
            tok = s.token
            if tok in self._val_tokens:
                val.append(s)
            elif tok in self._train_tokens:
                train.append(s)

        return train, val

    def get_train_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        train, _ = self._split(scenarios)
        return train

    def get_val_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        _, val = self._split(scenarios)
        return val

    def get_test_samples(self, scenarios: List[AbstractScenario], worker: WorkerPool) -> List[AbstractScenario]:
        return []
