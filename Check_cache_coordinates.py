#!/usr/bin/env python3
"""
src/utils/diagnose_coordinates.py

Drop this into _step to figure out exactly what frame every tensor is in.

Usage in pluto_trainer.py _step():

from src.utils.diagnose_coordinates import diagnose_coordinates

def _step(self, batch, prefix):
    features, targets, scenarios = batch
    data = features["feature"].data

    if self.global_step == 0 and prefix == "train":
        res = self.forward(data)
        diagnose_coordinates(data, res, self.history_steps)
        raise SystemExit("Diagnostic complete — check output above, then remove this block")

    res = self.forward(data)
    ...
"""

from __future__ import annotations

import torch


def _stat(t: torch.Tensor, name: str) -> None:
    t = t.float()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        print(f"  {name}: ALL NON-FINITE (nan/inf)")
        return

    print(
        f"  {name}: shape={tuple(t.shape)}  "
        f"min={finite.min().item():.3f}  "
        f"max={finite.max().item():.3f}  "
        f"absmax={finite.abs().max().item():.3f}  "
        f"mean={finite.mean().item():.3f}"
    )


def diagnose_coordinates(data: dict, res: dict, history_steps: int = 21) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("  COORDINATE FRAME DIAGNOSTIC")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Origin / angle
    # ------------------------------------------------------------------
    print("\n[1] STORED ORIGIN & ANGLE")
    if "origin" in data:
        _stat(data["origin"], "origin (UTM)")
        print(f"      origin[0] = {data['origin'][0].tolist()}")
        print("      → if > 1000: cache is in UTM, normalize_batch needed")
        print("      → if < 100:  already ego-relative, normalize_batch NOT needed")
    else:
        print("  origin: MISSING")

    if "angle" in data:
        _stat(data["angle"], "angle (rad)")
    else:
        print("  angle: MISSING")

    # ------------------------------------------------------------------
    # 2. Agent positions
    # ------------------------------------------------------------------
    print("\n[2] AGENT POSITIONS  (should all be < 150m if ego-relative)")
    pos = data["agent"]["position"]   # (B, A, T, 2)
    _stat(pos, "all agents position")
    _stat(pos[:, 0], "ego position (agent 0)")
    if pos.shape[1] > 1:
        _stat(pos[:, 1:], "other agents position")

    print(f"\n  ego position at present step (idx {history_steps - 1}):")
    print(f"    {pos[0, 0, history_steps - 1, :].tolist()}  ← should be [0,0] if normalized")

    # ------------------------------------------------------------------
    # 3. Agent targets
    # ------------------------------------------------------------------
    print("\n[3] AGENT TARGETS  (should be < 50m — relative to each agent at hist_steps-1)")
    if "target" in data["agent"]:
        tgt = data["agent"]["target"]   # (B, A, T_future, 3)
        _stat(tgt, "all agents target")
        _stat(tgt[:, 0], "ego target")
        if tgt.shape[1] > 1:
            _stat(tgt[:, 1:], "other agents target")

        print(f"\n  ego target[0] first 3 steps: {tgt[0, 0, :3, :2].tolist()}")
        print("  → if ego target absmax < 50: target is in per-agent-relative frame ✓")
        print("  → if ego target absmax > 100: target is in wrong frame ✗")
    else:
        print("  target: MISSING from data['agent']")

    # ------------------------------------------------------------------
    # 4. Model outputs
    # ------------------------------------------------------------------
    print("\n[4] MODEL OUTPUT TRAJECTORIES  (should be < 50m — ego-relative)")
    traj = res["trajectory"]   # (B, 1, Ka, T, 6)
    B, R, Ka, T, _ = traj.shape
    traj_xy = traj[..., :2]
    _stat(traj_xy, f"trajectory xy  (B={B}, R={R}, Ka={Ka}, T={T})")
    print(f"  traj[0,0,0,:3,:2] = {traj_xy[0, 0, 0, :3, :].tolist()}")

    print("\n[5] MODEL OUTPUT PREDICTIONS  (should be < 150m — ego-relative)")
    pred = res["prediction"]   # (B, A-1, T, 2)
    _stat(pred[..., :2], "prediction xy")
    print(f"  pred[0,0,:3,:2] = {pred[0, 0, :3, :2].tolist()}")

    # ------------------------------------------------------------------
    # 5. Frame comparison: are trajectory and ego_target in same frame?
    # ------------------------------------------------------------------
    print("\n[6] FRAME MATCH CHECK")
    if "target" in data["agent"]:
        ego_tgt = data["agent"]["target"][:, 0, :, :2]   # (B, T_future, 2)
        traj_flat = traj[:, 0, :, :, :2]                 # (B, Ka, T, 2)

        T_min = min(traj_flat.shape[2], ego_tgt.shape[1])
        endpoint_dist = torch.norm(
            traj_flat[:, :, T_min - 1, :] - ego_tgt[:, None, T_min - 1, :],
            dim=-1,
        )  # (B, Ka)
        min_dist = endpoint_dist.min(dim=-1)[0]   # (B,)
        print(f"  min endpoint dist (traj vs ego_target): {min_dist.mean().item():.3f}m")
        print("  → if < 5m:  trajectory and ego_target are in the SAME frame ✓")
        print("  → if > 20m: trajectory and ego_target are in DIFFERENT frames ✗")

    # ------------------------------------------------------------------
    # 6. Are prediction and prediction_target in same frame?
    # ------------------------------------------------------------------
    print("\n[7] PREDICTION FRAME MATCH CHECK")
    if "target" in data["agent"] and data["agent"]["target"].shape[1] > 1:
        pred_tgt = data["agent"]["target"][:, 1:, :, :2]  # (B, A-1, T_future, 2)
        pred_xy = res["prediction"][..., :2]              # (B, A-1, T, 2)

        T_min = min(pred_tgt.shape[2], pred_xy.shape[2])
        diff = torch.norm(pred_xy[:, :, :T_min] - pred_tgt[:, :, :T_min], dim=-1)

        vmask = data["agent"]["valid_mask"][:, 1:, history_steps:history_steps + T_min]
        masked_diff = (diff * vmask.float()).sum() / (vmask.float().sum() + 1e-6)

        print(f"  mean pred displacement vs agent target: {masked_diff.item():.3f}m")
        print("  → if < 5m:  prediction and agent_target are in the SAME frame ✓")
        print("  → if > 20m: prediction and agent_target are in DIFFERENT frames ✗")

        agent_pos_future = data["agent"]["position"][:, 1:, history_steps:history_steps + T_min, :2]
        diff2 = torch.norm(pred_xy[:, :, :T_min] - agent_pos_future, dim=-1)
        masked_diff2 = (diff2 * vmask.float()).sum() / (vmask.float().sum() + 1e-6)

        print(f"  mean pred displacement vs agent position (ego-relative): {masked_diff2.item():.3f}m")
        print("  → whichever is smaller tells you which frame prediction is in")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  SUMMARY — read these lines")
    print(sep)

    if "origin" in data:
        utm = data["origin"].abs().max().item()
        print(f"  origin magnitude: {utm:.1f}")
        if utm > 1000:
            print("  → DATA IS IN UTM FRAME — normalize_batch is needed")
        else:
            print("  → DATA IS ALREADY EGO-RELATIVE — normalize_batch is NOT needed")

    if "target" in data["agent"]:
        ego_tgt_max = data["agent"]["target"][:, 0].abs().max().item()
        print(f"  ego target absmax: {ego_tgt_max:.3f}")
        if ego_tgt_max > 100:
            print("  → ego target is in WRONG frame")
        else:
            print("  → ego target looks correct")

        if data["agent"]["target"].shape[1] > 1:
            other_tgt_max = data["agent"]["target"][:, 1:].abs().max().item()
            print(f"  other agents target absmax: {other_tgt_max:.3f}")
            if other_tgt_max > 100:
                print("  → other agent targets are in WRONG frame (per-agent-relative vs ego-relative mismatch)")
            else:
                print("  → other agent targets look correct")

    traj_max = res["trajectory"][..., :2].abs().max().item()
    print(f"  trajectory absmax: {traj_max:.3f}")
    if traj_max > 100:
        print("  → trajectory output is in WRONG frame")
    else:
        print("  → trajectory output looks correct")

    print(sep)
    print("  END DIAGNOSTIC")
    print(f"{sep}\n")
