#!/usr/bin/env python3
"""
Split a scenarios.yaml (nuPlan ScenarioFilter config) into two files,
each containing a subset of scenario_tokens.

Example:
    python split_scenarios_yaml.py \
        --input scenarios_400k.yaml \
        --out1 scenarios_200k_part1.yaml \
        --out2 scenarios_200k_part2.yaml \
        --chunk_size 200000 \
        --key scenario_tokens
"""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to original scenarios.yaml (≈400k tokens).")
    parser.add_argument("--out1", type=str, required=True,
                        help="Output path for first chunk yaml.")
    parser.add_argument("--out2", type=str, required=True,
                        help="Output path for second chunk yaml.")
    parser.add_argument("--chunk_size", type=int, default=200000,
                        help="Number of tokens in the first chunk (default: 200000).")
    parser.add_argument("--key", type=str, default="scenario_tokens",
                        help="YAML key containing the token list (default: scenario_tokens).")
    args = parser.parse_args()

    input_path = Path(args.input)
    out1_path = Path(args.out1)
    out2_path = Path(args.out2)

    if not input_path.exists():
        raise FileNotFoundError(f"Input yaml not found: {input_path}")

    with input_path.open("r") as f:
        data = yaml.safe_load(f)

    if args.key not in data:
        raise KeyError(f"Key '{args.key}' not found in {input_path}")

    tokens = data[args.key]
    if not isinstance(tokens, list):
        raise TypeError(f"Expected '{args.key}' to be a list, got {type(tokens)}")

    total = len(tokens)
    chunk_size = args.chunk_size

    if total < chunk_size:
        raise ValueError(
            f"Number of tokens ({total}) is smaller than chunk_size ({chunk_size})."
        )

    # First part: first `chunk_size` tokens
    tokens_part1 = tokens[:chunk_size]
    # Second part: the remaining tokens
    tokens_part2 = tokens[chunk_size:]

    print(f"Total tokens       : {total}")
    print(f"First yaml tokens  : {len(tokens_part1)}")
    print(f"Second yaml tokens : {len(tokens_part2)}")

    data1 = deepcopy(data)
    data2 = deepcopy(data)

    data1[args.key] = tokens_part1
    data2[args.key] = tokens_part2

    # Write out, preserving key order
    with out1_path.open("w") as f1:
        yaml.safe_dump(data1, f1, sort_keys=False)

    with out2_path.open("w") as f2:
        yaml.safe_dump(data2, f2, sort_keys=False)

    print(f"Wrote: {out1_path}")
    print(f"Wrote: {out2_path}")


if __name__ == "__main__":
    main()
