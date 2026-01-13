#!/usr/bin/env python3
"""
Compare two nuPlan ScenarioFilter YAML files and report overlapping scenario_tokens.

Usage:
  python compare_scenario_tokens.py \
      --yaml_a /path/to/my_train_scenarios.yaml \
      --yaml_b /path/to/val14_scenarios.yaml \
      [--out_common /path/to/common_tokens.txt]
"""

import argparse
import pathlib
from typing import Any, Dict, Iterable, Optional, Set

import yaml


def _find_scenario_tokens(node: Any) -> Optional[Iterable[Any]]:
    """
    Recursively search for a key 'scenario_tokens' in a nested dict/list structure.
    Return the first list it finds, or None.
    """
    if isinstance(node, dict):
        # Direct hit
        if "scenario_tokens" in node and isinstance(node["scenario_tokens"], list):
            return node["scenario_tokens"]
        # Recurse into values
        for v in node.values():
            res = _find_scenario_tokens(v)
            if res is not None:
                return res
    elif isinstance(node, list):
        for v in node:
            res = _find_scenario_tokens(v)
            if res is not None:
                return res
    return None


def load_tokens_from_yaml(path: pathlib.Path) -> Set[str]:
    """
    Load a YAML file and extract the scenario_tokens as a set of strings.
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)

    tokens_raw = _find_scenario_tokens(data)
    if tokens_raw is None:
        raise ValueError(f"No 'scenario_tokens' list found in {path}")

    tokens: Set[str] = set()
    for t in tokens_raw:
        # Normalize to plain string without surrounding whitespace/quotes
        s = str(t).strip().strip('"').strip("'")
        if s:
            tokens.add(s)

    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_a", type=str, required=True, help="First YAML file (e.g. your training scenarios)")
    parser.add_argument("--yaml_b", type=str, required=True, help="Second YAML file (e.g. val14 benchmark)")
    parser.add_argument(
        "--out_common",
        type=str,
        default=None,
        help="Optional path to write common tokens (one per line)",
    )

    args = parser.parse_args()

    path_a = pathlib.Path(args.yaml_a).expanduser().resolve()
    path_b = pathlib.Path(args.yaml_b).expanduser().resolve()

    if not path_a.exists():
        raise FileNotFoundError(f"{path_a} does not exist")
    if not path_b.exists():
        raise FileNotFoundError(f"{path_b} does not exist")

    tokens_a = load_tokens_from_yaml(path_a)
    tokens_b = load_tokens_from_yaml(path_b)

    common = sorted(tokens_a.intersection(tokens_b))

    print(f"File A: {path_a}")
    print(f"  #tokens: {len(tokens_a)}")
    print(f"File B: {path_b}")
    print(f"  #tokens: {len(tokens_b)}")
    print()
    print(f"# common scenario_tokens: {len(common)}")

    if not common:
        print("No overlap between the two YAMLs (good: train/val are disjoint).")
    else:
        print("Overlapping scenario_tokens:")
        for tok in common:
            print(f"  {tok}")

    if args.out_common and common:
        out_path = pathlib.Path(args.out_common).expanduser().resolve()
        with out_path.open("w") as f:
            for tok in common:
                f.write(tok + "\n")
        print()
        print(f"Common tokens written to: {out_path}")


if __name__ == "__main__":
    main()
