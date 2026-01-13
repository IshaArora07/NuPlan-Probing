#!/usr/bin/env python3
"""
Build a balanced EMoE subset from scene_labels*.jsonl.

- Keeps all samples of "common" classes.
- Randomly drops ~95% of specified rare classes (default: 4 and 5).
- Then clusters endpoints to produce anchors.

Outputs:
  1) scene_labels_balanced.jsonl         (full records, after downsampling)
  2) scene_labels_balanced_min.jsonl     (minimal: token + emoe_class_id)
  3) tokens_balanced.yaml                (YAML with scenario_tokens list)
  4) scene_anchors.npy                   (shape [num_classes, Ka, 2])
  5) summary.json                        (counts, skips, kmeans info)

Defaults:
- 6 classes (0..5)
- Downsample classes 4 and 5, keeping 5% each (removing ~95%).
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import yaml


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
# JSONL iterator
# -----------------------------
def iter_jsonl(paths: List[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    """Yield (path, line_no, record) for valid JSON lines; invalid lines yield a marker record."""
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
# Main pipeline
# -----------------------------
def run(
    input_paths: List[Path],
    output_dir: Path,
    num_classes: int,
    Ka: int,
    seed: int,
    minibatch_size: int,
    downsample_classes: List[int],
    keep_fraction: float,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    out_full = output_dir / "scene_labels_balanced.jsonl"
    out_min = output_dir / "scene_labels_balanced_min.jsonl"
    out_yaml = output_dir / "tokens_balanced.yaml"
    out_anchors = output_dir / "scene_anchors.npy"
    out_summary = output_dir / "summary.json"

    # Stats
    stats: Dict[str, Any] = {
        "inputs": [str(p) for p in input_paths],
        "num_classes": num_classes,
        "Ka": Ka,
        "seed": seed,
        "minibatch_size": minibatch_size,
        "downsample_classes": downsample_classes,
        "keep_fraction": keep_fraction,
        "counts_raw": {str(c): 0 for c in range(num_classes)},
        "counts_kept": {str(c): 0 for c in range(num_classes)},
        "skips": {
            "parse_error": 0,
            "missing_token": 0,
            "missing_class": 0,
            "class_out_of_range": 0,
            "downsampled": {str(c): 0 for c in range(num_classes)},
        },
        "endpoints": {
            "kept_with_endpoint": {str(c): 0 for c in range(num_classes)},
            "kept_missing_endpoint": {str(c): 0 for c in range(num_classes)},
        },
        "kmeans": {
            "algorithm": "MiniBatchKMeans",
            "per_class": {},
        },
    }

    endpoints_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    kept_tokens: List[str] = []

    with out_full.open("w") as f_full, out_min.open("w") as f_min:
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

            stats["counts_raw"][str(cid)] += 1

            # Downsampling for rare classes
            if cid in downsample_classes:
                if random.random() > keep_fraction:
                    stats["skips"]["downsampled"][str(cid)] += 1
                    continue

            # Keep this record
            stats["counts_kept"][str(cid)] += 1

            ep = extract_endpoint_xy(rec)
            if ep is not None:
                endpoints_by_class[cid].append(ep)
                stats["endpoints"]["kept_with_endpoint"][str(cid)] += 1
            else:
                stats["endpoints"]["kept_missing_endpoint"][str(cid)] += 1

            f_full.write(json.dumps(rec) + "\n")
            f_min.write(
                json.dumps({"token": token, "emoe_class_id": int(cid)}) + "\n"
            )
            kept_tokens.append(token)

    # -----------------------------
    # KMeans anchors per class
    # -----------------------------
    anchors = np.zeros((num_classes, Ka, 2), dtype=np.float32)

    for c in range(num_classes):
        pts = np.asarray(endpoints_by_class[c], dtype=np.float32)  # [N, 2]
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
            anchors[c, used_K:, :] = np.repeat(centers[:1, :], Ka - used_K, axis=0)
            kinfo["note"] = (
                f"filled remaining {Ka - used_K} anchors by repeating first center"
            )

        stats["kmeans"]["per_class"][str(c)] = kinfo

    np.save(out_anchors, anchors)

    # -----------------------------
    # YAML with scenario tokens
    # -----------------------------
    with out_yaml.open("w") as f_yaml:
        yaml.safe_dump({"scenario_tokens": kept_tokens}, f_yaml)

    # -----------------------------
    # Final summary
    # -----------------------------
    stats["outputs"] = {
        "scene_labels_full": str(out_full),
        "scene_labels_min": str(out_min),
        "tokens_yaml": str(out_yaml),
        "anchors_npy": str(out_anchors),
        "summary_json": str(out_summary),
    }
    stats["anchors_shape"] = list(anchors.shape)

    with out_summary.open("w") as f_sum:
        json.dump(stats, f_sum, indent=2)

    # Console summary
    print("\n[DONE] Wrote outputs to:", output_dir)
    print("Raw vs kept counts per class:")
    for c in range(num_classes):
        raw = stats["counts_raw"][str(c)]
        kept = stats["counts_kept"][str(c)]
        print(
            f"  class {c}: raw={raw}, kept={kept}, "
            f"downsampled={stats['skips']['downsampled'][str(c)]}"
        )
    print("Anchors saved:", out_anchors, "shape", anchors.shape)
    print("Summary:", out_summary)
    print("Tokens YAML:", out_yaml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files (e.g. scene_labels_part1.jsonl scene_labels_part2.jsonl ...)",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of classes (0..num_classes-1)",
    )
    parser.add_argument(
        "--Ka",
        type=int,
        default=24,
        help="Number of anchors per class",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=8192)
    parser.add_argument(
        "--downsample_classes",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Class IDs to downsample (default: 4 5)",
    )
    parser.add_argument(
        "--keep_fraction",
        type=float,
        default=0.05,
        help="Fraction to keep for downsampled classes (default: 0.05 ~ keep 5%)",
    )

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
        Ka=int(args.Ka),
        seed=int(args.seed),
        minibatch_size=int(args.minibatch_size),
        downsample_classes=list(args.downsample_classes),
        keep_fraction=float(args.keep_fraction),
    )


if __name__ == "__main__":
    main()
