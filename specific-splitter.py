# src/splitters/token_list_splitter.py

from typing import List, Set
from nuplan.planning.training.data_splitter.data_splitter import DataSplitter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario


class TokenListSplitter(DataSplitter):
    """
    Deterministic splitter based on explicit scenario tokens.

    - Scenarios whose token is in `val_tokens` → validation
    - Scenarios whose token is in `train_tokens` → training
    - Everything else is ignored (not used by either split)
    """

    def __init__(
        self,
        train_tokens: List[str],
        val_tokens: List[str],
    ):
        self._train_tokens: Set[str] = set(train_tokens)
        self._val_tokens: Set[str] = set(val_tokens)

        # Safety check
        overlap = self._train_tokens.intersection(self._val_tokens)
        if overlap:
            raise ValueError(f"Train/val token overlap detected: {len(overlap)} tokens")

    def get_train_scenarios(
        self, scenarios: List[AbstractScenario]
    ) -> List[AbstractScenario]:
        return [
            s for s in scenarios
            if s.token in self._train_tokens
        ]

    def get_validation_scenarios(
        self, scenarios: List[AbstractScenario]
    ) -> List[AbstractScenario]:
        return [
            s for s in scenarios
            if s.token in self._val_tokens
        ]
