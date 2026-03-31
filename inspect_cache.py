#!/usr/bin/env python3
"""
inspect_agent_outliers.py

Finds which agents in features.gz have large (UTM-like) position values
and checks whether they are valid agents or just invalid padding.

Run:
    python inspect_agent_outliers.py --cache_path /your/cache/path
"""

import argparse
import gzip
import os
import pickle

import numpy as np


def inspect(cache_path: str) -> None:
    sep = "=" * 65

    # ── Find first features.gz ────────────────────────────────
    fpath = None
    for root, _, files in os.walk(cache_path):
        for fname in files:
            if fname == "features.gz":
                fpath = os.path.join(root, fname)
                break
        if fpath:
            break

    if fpath is None:
        print(f"No features.gz found under: {cache_path}")
        return

    print(f"File: {fpath}\n")

    with gzip.open(fpath, "rb") as fh:
        raw = pickle.load(fh)

    # ── Unwrap to dict ────────────────────────────────────────
    if isinstance(raw, dict):
        d = raw
    elif hasattr(raw, "data") and isinstance(raw.data, dict):
        d = raw.data
    elif hasattr(raw, "__dict__"):
        d = raw.__dict__
        if "data" in d and isinstance(d["data"], dict):
            d = d["data"]
    else:
        print("Unknown structure")
        return

    pos = d["agent"]["position"]       # (A, T, 2)
    vmask = d["agent"]["valid_mask"]   # (A, T)
    hdg = d["agent"]["heading"]        # (A, T)
    tgt = d["agent"].get("target")     # (A, T_future, 3) or None

    A, T, _ = pos.shape
    present_idx = min(20, T - 1)
    future_idx = min(21, T - 1)

    print(sep)
    print(f"AGENT POSITION ANALYSIS  A={A}, T={T}")
    print(sep)

    # Per-agent absmax
    agent_absmax = np.abs(pos).max(axis=(1, 2))   # (A,)
    large_agents = np.where(agent_absmax > 100)[0]
    small_agents = np.where(agent_absmax <= 100)[0]

    print(f"\nAgents with absmax > 100m : {len(large_agents)}")
    print(f"Agents with absmax <= 100m: {len(small_agents)}")

    print("\nAll agent absmax values:")
    for a in range(A):
        valid_steps = int(vmask[a].sum())
        print(
            f"  agent {a:2d}: absmax={agent_absmax[a]:12.3f}  "
            f"valid_steps={valid_steps}/{T}  "
            f"pos_at_present={pos[a, present_idx, :].tolist()}"
        )

    # ── Deep dive on large agents ─────────────────────────────
    print(f"\n{sep}")
    print("DEEP DIVE: agents with absmax > 100m")
    print(sep)

    for a in large_agents[:5]:
        print(f"\n  agent {a}:")
        print(f"    valid_mask any       : {bool(vmask[a].any())}")
        print(f"    valid_mask sum       : {int(vmask[a].sum())}")
        print(f"    valid at present     : {bool(vmask[a, present_idx])}")
        print(f"    pos at t=0           : {pos[a, 0, :]}")
        print(f"    pos at present       : {pos[a, present_idx, :]}")
        print(f"    pos at future step   : {pos[a, future_idx, :]}")
        print(f"    heading at present   : {hdg[a, present_idx]:.4f}")
        print(f"    pos absmax           : {agent_absmax[a]:.3f}")

        step_absmax = np.abs(pos[a]).max(axis=1)  # (T,)
        large_steps = np.where(step_absmax > 100)[0]
        print(f"    timesteps absmax >100: {large_steps[:10].tolist()}")

        if len(large_steps) > 0:
            t = large_steps[0]
            print(f"    first large step t={t}: {pos[a, t, :]}")
            print(f"    valid at that step    : {bool(vmask[a, t])}")

    # ── Check if large values are always invalid ──────────────
    print(f"\n{sep}")
    print("VALIDITY CHECK: are large positions always masked invalid?")
    print(sep)

    pos_flat = pos.reshape(-1, 2)
    vmask_flat = vmask.reshape(-1).astype(bool)

    large_pos_mask = np.abs(pos_flat).max(axis=1) > 100
    valid_and_large = large_pos_mask & vmask_flat
    inval_and_large = large_pos_mask & ~vmask_flat

    print(f"  positions absmax > 100m              : {int(large_pos_mask.sum())}")
    print(f"  marked VALID   (bad — UTM leak)      : {int(valid_and_large.sum())}")
    print(f"  marked INVALID (ok — padding)        : {int(inval_and_large.sum())}")

    if valid_and_large.sum() == 0:
        print("\n  ✅ ALL large positions are invalid/masked")
        print("     These UTM-like values are padding only.")
        print("     valid_mask correctly excludes them.")
        print("\n  CONCLUSION: data IS normalized.")
        print("  Do NOT use normalize_batch.")
        print("  Metric bug is likely elsewhere.")
    else:
        print(f"\n  ❌ {int(valid_and_large.sum())} large positions are VALID")
        print("     Genuine UTM coordinates leaked into valid slots.")
        print("     normalize_batch IS needed.")

        bad_indices = np.where(valid_and_large)[0][:3]
        for idx in bad_indices:
            a, t = divmod(int(idx), T)
            print(
                f"     example: agent={a}, t={t}, "
                f"pos={pos[a, t, :]}, valid={bool(vmask[a, t])}"
            )

    # ── Target check ─────────────────────────────────────────
    print(f"\n{sep}")
    print("TARGET CHECK")
    print(sep)

    if tgt is not None:
        print(f"  target shape  : {tgt.shape}")
        print(f"  target absmax : {np.abs(tgt).max():.4f}")
        print(f"  target first4 : {tgt.reshape(-1)[:4].tolist()}")

        tgt_agent_absmax = np.abs(tgt).max(axis=(1, 2))
        large_tgt = np.where(tgt_agent_absmax > 80)[0]

        print(f"  agents with target absmax > 80: {large_tgt.tolist()}")

        for a in large_tgt[:3]:
            print(
                f"    agent {a}: tgt absmax={tgt_agent_absmax[a]:.3f}  "
                f"valid_any={bool(vmask[a].any())}"
            )
    else:
        print("  target: NOT PRESENT in cache")

    print(f"\n{sep}")
    print("END")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to nuPlan cache directory",
    )
    args = parser.parse_args()
    inspect(args.cache_path)
