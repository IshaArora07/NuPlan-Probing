#!/usr/bin/env python3
"""
Build a 200k EMoE subset from multiple scene_labels*.jsonl files, with:

  - Classes 4 and 5: keep ALL samples (no downsampling).
  - Classes 0,1,2,3: fill the remaining budget equally (as much as possible).

Outputs:
  1) scene_labels_200k.jsonl        (full records, filtered subset)
  2) scene_labels_200k_min.jsonl    (minimal: token + emoe_class_id)
  3) tokens_200k.txt                (one token per line; selected set)
  4) scene_anchors_200k.npy         (shape [num_classes, Ka, 2])
  5) summary_200k.json              (counts, selection, kmeans info)
  6) scenarios_200k.yaml            (scenario_tokens: ... for nuPlan filters)

Assumptions:
- 6 classes (class ids 0..5). No roundabout (class 6) here.
- Each input line is JSON with at least:
    "token" or "scenario_token" or "scenario_id"
    "emoe_class_id" (0..5)
- Endpoint can be stored under several possible keys; we try:
    - endpoint_xy
    - endpoint
    - endpoint_rel
    - debug.endpoint_xy
    - debug.endpoint
    - debug.endpoint_rel
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


# -----------------------------
# Helpers: safe extraction
# -----------------------------

def _get(d: Dict[str, Any], path: str) -> Any:
    """Get nested field with dotted path, returns None if missing."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def extract_token(rec: Dict[str, Any]) -> Optional[str]:
    for k in ["token", "scenario_token", "scenario_id"]:
        v = rec.get(k, None)
        if v is not None and str(v).strip():
            return str(v)
    return None


def extract_class_id(rec: Dict[str, Any]) -> Optional[int]:
    for k in ["emoe_class_id", "class_id", "scene_class_id"]:
        if k in rec:
            try:
                return int(rec[k])
            except Exception:
                return None
    return None


def extract_endpoint_xy(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Try a few likely endpoint keys. Returns np.ndarray shape (2,) float32 or None.
    """
    candidates = [
        "endpoint_xy",
        "endpoint",
        "endpoint_rel",
        "debug.endpoint_xy",
        "debug.endpoint",
        "debug.endpoint_rel",
    ]
    for key in candidates:
        v = _get(rec, key) if "." in key else rec.get(key, None)
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if arr.shape[0] != 2:
            continue
        if not np.all(np.isfinite(arr)):
            continue
        return arr.astype(np.float32)
    return None


def iter_jsonl(paths: List[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    """Yield (path, line_no, record) for valid JSON lines; invalid lines yield a small error-marked record."""
    for p in paths:
        with p.open("r") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield p, i, rec
                    else:
                        yield p, i, {"__parse_error__": "json_not_object"}
                except Exception as e:
                    yield p, i, {"__parse_error__": f"json_decode_error: {type(e).__name__}"}


# -----------------------------
# Selection logic (new part)
# -----------------------------

def select_200k_subset(
    records_by_class: Dict[int, List[Dict[str, Any]]],
    num_classes: int,
    total_target: int,
    fixed_classes: List[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    - Keep ALL samples of classes in `fixed_classes` (e.g. [4, 5]).
    - Fill remaining budget up to `total_target` with classes in {0..num_classes-1}\fixed_classes,
      as evenly as possible.
    - Returns a shuffled list of selected records.
    """

    rng = np.random.RandomState(seed)

    # 1) Take all fixed classes
    selected: List[Dict[str, Any]] = []
    fixed_total = 0
    for c in fixed_classes:
        cls_recs = records_by_class.get(c, [])
        selected.extend(cls_recs)
        fixed_total += len(cls_recs)

    if fixed_total > total_target:
        raise ValueError(
            f"Fixed classes {fixed_classes} already exceed target: "
            f"{fixed_total} > {total_target}"
        )

    remaining = total_target - fixed_total
    other_classes = [c for c in range(num_classes) if c not in fixed_classes]

    # 2) Distribute remaining equally across other classes (0,1,2,3)
    if remaining <= 0 or not other_classes:
        # Nothing more to add
        rng.shuffle(selected)
        return selected

    per_class_target = remaining // len(other_classes)
    leftover = remaining % len(other_classes)

    # First pass: base allocation
    desired_per_class: Dict[int, int] = {}
    for i, c in enumerate(sorted(other_classes)):
        desired = per_class_target + (1 if i < leftover else 0)
        available = len(records_by_class.get(c, []))
        desired_per_class[c] = min(desired, available)

    # Optional second pass: redistribute any unfilled quota to classes that still have spare
    assigned = sum(desired_per_class.values())
    unfilled = remaining - assigned

    if unfilled > 0:
        # Try to give extra to classes with spare samples
        for c in sorted(other_classes):
            if unfilled <= 0:
                break
            spare = len(records_by_class.get(c, [])) - desired_per_class[c]
            if spare <= 0:
                continue
            add = min(spare, unfilled)
            desired_per_class[c] += add
            unfilled -= add
        # If still unfilled > 0, we just end up with fewer than total_target
        # This is fine; it means there isn't enough data.

    # 3) Sample for other classes according to desired_per_class
    for c in sorted(other_classes):
        cls_recs = records_by_class.get(c, [])
        if not cls_recs:
            continue
        # shuffle in-place then take desired slice
        idx = rng.permutation(len(cls_recs))
        take = desired_per_class[c]
        if take > 0:
            selected.extend([cls_recs[i] for i in idx[:take]])

    rng.shuffle(selected)
    return selected


# -----------------------------
# Main pipeline
# -----------------------------

def run(
    input_paths: List[Path],
    output_dir: Path,
    num_classes: int,
    total_target: int,
    fixed_classes: List[int],
    Ka: int,
    seed: int,
    minibatch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_full = output_dir / "scene_labels_200k.jsonl"
    out_min = output_dir / "scene_labels_200k_min.jsonl"
    out_tokens = output_dir / "tokens_200k.txt"
    out_anchors = output_dir / "scene_anchors_200k.npy"
    out_summary = output_dir / "summary_200k.json"
    out_scenarios_yaml = output_dir / "scenarios_200k.yaml"

    # Stats container
    stats: Dict[str, Any] = {
        "inputs": [str(p) for p in input_paths],
        "num_classes": num_classes,
        "total_target": total_target,
        "fixed_classes": fixed_classes,
        "Ka": Ka,
        "seed": seed,
        "minibatch_size": minibatch_size,
        "original_counts": {str(c): 0 for c in range(num_classes)},
        "selected_counts": {str(c): 0 for c in range(num_classes)},
        "skips": {
            "parse_error": 0,
            "missing_token": 0,
            "missing_class": 0,
            "class_out_of_range": 0,
            "duplicate_token": 0,
        },
        "endpoints": {
            "selected_with_endpoint": {str(c): 0 for c in range(num_classes)},
            "selected_missing_endpoint": {str(c): 0 for c in range(num_classes)},
        },
        "kmeans": {
            "algorithm": "MiniBatchKMeans",
            "per_class": {},
        },
    }

    # 1) First pass: read everything into memory, group by class, dedupe by token
    token_seen: Dict[str, Dict[str, Any]] = {}
    class_for_token: Dict[str, int] = {}

    for path, line_no, rec in iter_jsonl(input_paths):
        if "__parse_error__" in rec:
            stats["skips"]["parse_error"] += 1
            continue

        token = extract_token(rec)
        if token is None:
            stats["skips"]["missing_token"] += 1
            continue

        cid = extract_class_id(rec)
        if cid is None:
            stats["skips"]["missing_class"] += 1
            continue
        if cid < 0 or cid >= num_classes:
            stats["skips"]["class_out_of_range"] += 1
            continue

        # If token already seen, last one wins
        if token in token_seen:
            stats["skips"]["duplicate_token"] += 1

        token_seen[token] = rec
        class_for_token[token] = cid

    # Build records_by_class
    records_by_class: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(num_classes)}
    for token, rec in token_seen.items():
        cid = class_for_token[token]
        records_by_class[cid].append(rec)
        stats["original_counts"][str(cid)] += 1

    # 2) Selection: fixed classes (4,5) + balanced 0-3 to reach ~200k
    selected_records: List[Dict[str, Any]] = select_200k_subset(
        records_by_class=records_by_class,
        num_classes=num_classes,
        total_target=total_target,
        fixed_classes=fixed_classes,
        seed=seed,
    )

    # Count selected per class
    for rec in selected_records:
        cid = int(rec["emoe_class_id"])
        stats["selected_counts"][str(cid)] += 1

    # 3) From selected subset, collect endpoints per class for KMeans
    endpoints_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    kept_tokens: List[str] = []

    # And simultaneously write out JSONLs and tokens.txt
    with out_full.open("w") as f_full, out_min.open("w") as f_min, out_tokens.open("w") as f_tok:
        for rec in selected_records:
            token = extract_token(rec)
            cid = extract_class_id(rec)
            if token is None or cid is None or cid < 0 or cid >= num_classes:
                # Should not happen at this point, but be safe
                continue

            ep = extract_endpoint_xy(rec)
            if ep is not None:
                endpoints_by_class[cid].append(ep)
                stats["endpoints"]["selected_with_endpoint"][str(cid)] += 1
            else:
                stats["endpoints"]["selected_missing_endpoint"][str(cid)] += 1

            f_full.write(json.dumps(rec) + "\n")
            f_min.write(json.dumps({"token": token, "emoe_class_id": int(cid)}) + "\n")
            f_tok.write(token + "\n")
            kept_tokens.append(token)

    # 4) KMeans anchors per class (MiniBatchKMeans)
    anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)

    for c in range(num_classes):
        pts_list = endpoints_by_class[c]
        if len(pts_list) == 0:
            stats["kmeans"]["per_class"][str(c)] = {
                "n_points": 0,
                "requested_Ka": Ka,
                "used_K": 0,
                "note": "no endpoints; anchors left as zeros",
            }
            continue

        pts = np.asarray(pts_list, dtype=np.float32)  # [N,2]
        n_pts = int(pts.shape[0])
        used_K = min(Ka, n_pts)

        kinfo: Dict[str, Any] = {
            "n_points": n_pts,
            "requested_Ka": Ka,
            "used_K": used_K,
            "note": "",
        }

        kmeans = MiniBatchKMeans(
            n_clusters=used_K,
            random_state=seed,
            batch_size=minibatch_size,
            n_init="auto" if hasattr(MiniBatchKMeans, "n_init") else 10,
            max_no_improvement=20,
            reassignment_ratio=0.01,
        )
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_.astype(np.float32)  # [used_K,2]

        anchors[c, :used_K, :] = centers
        if used_K < Ka:
            anchors[c, used_K:, :] = np.repeat(centers[:1, :], Ka - used_K, axis=0)
            kinfo["note"] = f"filled remaining {Ka-used_K} anchors by repeating first center"

        stats["kmeans"]["per_class"][str(c)] = kinfo

    np.save(out_anchors, anchors)

    # 5) Write scenarios_200k.yaml for nuPlan scenario_filter
    with out_scenarios_yaml.open("w") as f_yaml:
        f_yaml.write("scenario_tokens:\n")
        for t in kept_tokens:
            f_yaml.write(f'  - "{t}"\n')

    # 6) Final summary JSON
    stats["outputs"] = {
        "scene_labels_full": str(out_full),
        "scene_labels_min": str(out_min),
        "tokens": str(out_tokens),
        "anchors_npy": str(out_anchors),
        "scenarios_yaml": str(out_scenarios_yaml),
        "summary_json": str(out_summary),
    }
    stats["anchors_shape"] = list(anchors.shape)

    with out_summary.open("w") as f_sum:
        json.dump(stats, f_sum, indent=2)

    # Console print
    print("\n[DONE] Wrote outputs to:", output_dir)
    print("Original counts per class:")
    for c in range(num_classes):
        print(f"  class {c}: {stats['original_counts'][str(c)]}")
    print("Selected counts per class:")
    for c in range(num_classes):
        print(f"  class {c}: {stats['selected_counts'][str(c)]}")
    print("Total selected:", sum(stats["selected_counts"].values()))
    print("Anchors saved:", out_anchors, "shape", anchors.shape)
    print("Summary:", out_summary)
    print("Scenarios YAML:", out_scenarios_yaml)


def main():
    parser = argparse.ArgumentParser(
        description="Build a 200k EMoE subset (classes 4,5 fully kept; others balanced), "
                    "compute anchors, and export configs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files (e.g. scene_labels_part1.jsonl scene_labels_part2.jsonl ...)",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes (0..num_classes-1).")
    parser.add_argument(
        "--total_target",
        type=int,
        default=200_000,
        help="Total target number of samples (approx).",
    )
    parser.add_argument(
        "--fixed_classes",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Classes to include fully (no downsampling), e.g. 4 5.",
    )
    parser.add_argument("--Ka", type=int, default=24, help="Anchors per class.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=8192)

    args = parser.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    run(
        input_paths=input_paths,
        output_dir=output_dir,
        num_classes=int(args.num_classes),
        total_target=int(args.total_target),
        fixed_classes=list(args.fixed_classes),
        Ka=int(args.Ka),
        seed=int(args.seed),
        minibatch_size=int(args.minibatch_size),
    )


if __name__ == "__main__":
    main()
