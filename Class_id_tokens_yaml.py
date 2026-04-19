#!/usr/bin/env python3
"""
Extract scenario tokens for a given EMoE class ID from scene_labels.jsonl
and write them to a YAML file in the format:

scenario_tokens:
- e3e2933994835eba
- '3069f8795e1c5116'
- f12b46915ede5842
- …

Usage:
python extract_tokens_by_class.py \
  --labels_path ./emoe_precomputed/scene_labels.jsonl \
  --class_id 3 \
  --output_path ./tokens_class3.yaml
"""

import json
import argparse
from pathlib import Path

EMOE_SCENE_TYPES = [
    "left_turn_at_intersection",   # 0
    "straight_at_intersection",    # 1
    "right_turn_at_intersection",  # 2
    "straight_non_intersection",   # 3
    "roundabout",                  # 4
    "u_turn",                      # 5
    "others",                      # 6
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to scene_labels.jsonl",
    )

    parser.add_argument(
        "--class_id",
        type=int,
        required=True,
        choices=range(7),
        help="EMoE class ID to extract (0-6)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output YAML file path. Defaults to tokens_class<id>.yaml in the same dir as labels_path.",
    )

    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Default output path
    if args.output_path is None:
        output_path = labels_path.parent / f"tokens_class{args.class_id}.yaml"
    else:
        output_path = Path(args.output_path)

    class_name = EMOE_SCENE_TYPES[args.class_id]

    print(f"[INFO] Reading: {labels_path}")
    print(f"[INFO] Filtering for class {args.class_id} ({class_name})")

    tokens = []
    total = 0

    with labels_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            if int(record["emoe_class_id"]) == args.class_id:
                tokens.append(str(record["token"]))

    print(f"[INFO] Total scenarios in file : {total}")
    print(f"[INFO] Matched (class {args.class_id}): {len(tokens)}")

    # Write YAML manually to match exact format
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(f"# EMoE class {args.class_id}: {class_name}\n")
        f.write(f"# Total tokens: {len(tokens)}\n")
        f.write("scenario_tokens:\n")

        for token in tokens:
            f.write(f"  - '{token}'\n")

    print(f"[INFO] Saved: {output_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
