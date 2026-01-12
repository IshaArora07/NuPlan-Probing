#!/usr/bin/env python3
"""
Generic splitter for a scenarios.yaml with a large list of scenario_tokens.

Example (split 1.2M tokens into 6 parts):
    python split_scenarios_yaml_generic.py \
        --input scenarios_1_2M.yaml \
        --out-prefix scenarios_200k \
        --key scenario_tokens \
        --num-splits 6
"""

import argparse
from copy import deepcopy

import yaml


# ----------------------------------------------------------------------
# Force quoted strings in YAML for our tokens
# ----------------------------------------------------------------------
class Quoted(str):
    """String that will always be dumped with double quotes in YAML."""
    pass


def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(Quoted, quoted_presenter)


# ----------------------------------------------------------------------
# Main split logic
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input scenarios.yaml")
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output files; script will append _part{i}.yaml",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="scenario_tokens",
        help="YAML key containing the list of tokens (default: scenario_tokens)",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        required=True,
        help="Number of splits (e.g. 6 for 1.2M -> 6 parts)",
    )
    args = parser.parse_args()

    # Load original YAML
    with open(args.input, "r") as f:
        data = yaml.safe_load(f)

    if args.key not in data:
        raise KeyError(f"Key '{args.key}' not found in {args.input}")

    tokens = data[args.key]
    if not isinstance(tokens, list):
        raise TypeError(f"Field '{args.key}' is not a list in {args.input}")

    total = len(tokens)
    if args.num_splits < 2:
        raise ValueError("--num-splits must be >= 2")
    if args.num_splits > total:
        raise ValueError(
            f"--num-splits ({args.num_splits}) cannot exceed total tokens ({total})"
        )

    # Compute split sizes as even as possible
    base = total // args.num_splits
    rem = total % args.num_splits

    split_sizes = []
    for i in range(args.num_splits):
        size = base + (1 if i < rem else 0)
        split_sizes.append(size)

    print(f"Total tokens: {total}")
    print(f"Num splits : {args.num_splits}")
    print("Split sizes:", split_sizes)

    # Perform splitting
    start_idx = 0
    for i, size in enumerate(split_sizes):
        end_idx = start_idx + size
        part_tokens = tokens[start_idx:end_idx]

        data_part = deepcopy(data)
        data_part[args.key] = [Quoted(str(t)) for t in part_tokens]

        out_path = f"{args.out_prefix}_part{i+1}.yaml"
        with open(out_path, "w") as f_out:
            yaml.safe_dump(data_part, f_out, sort_keys=False)

        print(
            f"Part {i+1}: indices [{start_idx}:{end_idx}) -> {len(part_tokens)} tokens -> {out_path}"
        )

        start_idx = end_idx


if __name__ == "__main__":
    main()
