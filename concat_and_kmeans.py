#!/usr/bin/env python3
"""
Concatenate multiple EMoE scene_labels*.jsonl files, cap to max N samples per class,
cluster trajectory endpoints to produce anchors, and output:
  1) scene_labels_200k.jsonl        (full records, filtered + capped)
  2) scene_labels_200k_min.jsonl    (minimal: token + class_id)
  3) tokens_200k.txt                (one token per line; selected set)
  4) scene_anchors.npy              (shape [6, Ka, 2])
  5) summary.json                   (counts, skips, kmeans info)

Assumptions:
- 6 classes (class ids 0..5). Roundabout removed.
- Each input line is JSON with at least:
    token (or scenario_token) and emoe_class_id
- Endpoint can be stored under a few possible keys; script tries common options:
    - endpoint_xy
    - endpoint
    - endpoint_rel
    - debug.endpoint_xy
    - debug.endpoint
    - debug.endpoint_rel

Sampling policy:
- simple first-come-first-served (stream order), per your request.

KMeans:
- Uses MiniBatchKMeans (much faster for large N). Produces Ka anchors per class.
"""

import argparse
import json
import math
import os
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
        # Accept list/tuple/np.ndarray with length 2
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


# -----------------------------
# Main pipeline
# -----------------------------
def iter_jsonl(paths: List[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    """Yield (path, line_no, record) for valid JSON lines; invalid lines yield empty record with error marker."""
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


def run(
    input_paths: List[Path],
    output_dir: Path,
    num_classes: int,
    max_per_class: int,
    Ka: int,
    seed: int,
    minibatch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_full = output_dir / f"scene_labels_{max_per_class//1000}k.jsonl"
    out_min = output_dir / f"scene_labels_{max_per_class//1000}k_min.jsonl"
    out_tokens = output_dir / f"tokens_{max_per_class//1000}k.txt"
    out_anchors = output_dir / "scene_anchors.npy"
    out_summary = output_dir / "summary.json"

    # Stats
    stats: Dict[str, Any] = {
        "inputs": [str(p) for p in input_paths],
        "num_classes": num_classes,
        "max_per_class": max_per_class,
        "Ka": Ka,
        "seed": seed,
        "minibatch_size": minibatch_size,
        "counts": {str(c): 0 for c in range(num_classes)},
        "skips": {
            "parse_error": 0,
            "missing_token": 0,
            "missing_class": 0,
            "class_out_of_range": 0,
            "cap_reached": 0,
        },
        "endpoints": {
            "kept_with_endpoint": {str(c): 0 for c in range(num_classes)},
            "kept_missing_endpoint": {str(c): 0 for c in range(num_classes)},
            "skipped_missing_endpoint_before_cap": {str(c): 0 for c in range(num_classes)},
        },
        "kmeans": {
            "algorithm": "MiniBatchKMeans",
            "per_class": {},
        },
    }

    # Storage (we only store endpoints for kept samples; full records streamed to file)
    endpoints_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    kept_tokens: List[str] = []

    # Keep counters per class
    kept_per_class = [0 for _ in range(num_classes)]

    # Stream and write
    with out_full.open("w") as f_full, out_min.open("w") as f_min, out_tokens.open("w") as f_tok:
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

            # cap check
            if kept_per_class[cid] >= max_per_class:
                stats["skips"]["cap_reached"] += 1
                continue

            # Keep this record
            kept_per_class[cid] += 1
            stats["counts"][str(cid)] += 1

            # endpoints (optional)
            ep = extract_endpoint_xy(rec)
            if ep is not None:
                endpoints_by_class[cid].append(ep)
                stats["endpoints"]["kept_with_endpoint"][str(cid)] += 1
            else:
                stats["endpoints"]["kept_missing_endpoint"][str(cid)] += 1

            # Write outputs
            f_full.write(json.dumps(rec) + "\n")
            f_min.write(json.dumps({"token": token, "emoe_class_id": int(cid)}) + "\n")
            f_tok.write(token + "\n")
            kept_tokens.append(token)

    # KMeans anchors per class
    anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)

    for c in range(num_classes):
        pts = np.asarray(endpoints_by_class[c], dtype=np.float32)  # [N,2]
        n_pts = int(pts.shape[0])

        kinfo: Dict[str, Any] = {
            "n_points": n_pts,
            "requested_Ka": Ka,
            "used_K": 0,
            "note": "",
        }

        if n_pts == 0:
            kinfo["note"] = "no endpoints available; anchors left as zeros"
            stats["kmeans"]["per_class"][str(c)] = kinfo
            continue

        used_K = min(Ka, n_pts)
        kinfo["used_K"] = used_K

        # MiniBatchKMeans (fast)
        kmeans = MiniBatchKMeans(
            n_clusters=used_K,
            random_state=seed,
            batch_size=minibatch_size,
            n_init="auto" if hasattr(MiniBatchKMeans, "n_init") else 10,
            max_no_improvement=20,
            reassignment_ratio=0.01,
        )
        kmeans.fit(pts)
        centers = kmeans.cluster_centers_.astype(np.float32)

        anchors[c, :used_K, :] = centers
        if used_K < Ka:
            # repeat the first center to fill
            anchors[c, used_K:, :] = np.repeat(centers[:1, :], Ka - used_K, axis=0)
            kinfo["note"] = f"filled remaining {Ka-used_K} anchors by repeating first center"

        stats["kmeans"]["per_class"][str(c)] = kinfo

    np.save(out_anchors, anchors)

    # Final summary
    stats["outputs"] = {
        "scene_labels_full": str(out_full),
        "scene_labels_min": str(out_min),
        "tokens": str(out_tokens),
        "anchors_npy": str(out_anchors),
        "summary_json": str(out_summary),
    }
    stats["anchors_shape"] = list(anchors.shape)

    with out_summary.open("w") as f_sum:
        json.dump(stats, f_sum, indent=2)

    # Print a brief console summary
    print("\n[DONE] Wrote outputs to:", output_dir)
    print("Counts per class:")
    for c in range(num_classes):
        print(
            f"  class {c}: kept={stats['counts'][str(c)]} "
            f"(with_ep={stats['endpoints']['kept_with_endpoint'][str(c)]}, "
            f"missing_ep={stats['endpoints']['kept_missing_endpoint'][str(c)]})"
        )
    print("Anchors saved:", out_anchors, "shape", anchors.shape)
    print("Summary:", out_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files (e.g. scene_labels_part1.jsonl scene_labels_part2.jsonl ...)",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=6, help="6 classes (0..5), roundabout removed")
    parser.add_argument("--max_per_class", type=int, default=200000)
    parser.add_argument("--Ka", type=int, default=24)
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
        max_per_class=int(args.max_per_class),
        Ka=int(args.Ka),
        seed=int(args.seed),
        minibatch_size=int(args.minibatch_size),
    )


if __name__ == "__main__":
    main()
