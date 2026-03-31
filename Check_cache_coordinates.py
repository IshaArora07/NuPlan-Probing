#!/usr/bin/env python3
"""
src/utils/diagnose_coordinates.py

Comprehensive diagnostic to determine exactly what coordinate frame
every tensor is in, and draw conclusions about what fix is needed.

Usage in pluto_trainer.py _step():

from src.utils.diagnose_coordinates import diagnose_coordinates

def _step(self, batch, prefix):
    features, targets, scenarios = batch
    data = features["feature"].data

    if self.global_step == 0 and prefix == "train":
        res = self.forward(data)
        diagnose_coordinates(data, res, self.history_steps)
        raise SystemExit("Diagnostic complete — remove this block and fix accordingly")

    res = self.forward(data)
    losses = self._compute_objectives(res, data, prefix=prefix)
    metrics = self._compute_metrics(res, data, prefix)
    self._log_step(losses["loss"], losses, metrics, prefix)
    return losses["loss"] if self.training else 0.0
"""

from __future__ import annotations

import torch

SEP = "=" * 65
SEP2 = "-" * 65


def _absmax(t: torch.Tensor) -> float:
    return t.float().abs().max().item()


def _conclude(label: str, value: float, small_thresh: float = 50.0) -> str:
    if value < small_thresh:
        return f"  ✅ {label}: {value:.3f}  → ego-relative / correct"
    return f"  ❌ {label}: {value:.3f}  → WRONG FRAME or unnormalized"


def diagnose_coordinates(data: dict, res: dict, history_steps: int = 21) -> None:
    print(f"\n{SEP}")
    print("  DEEP COORDINATE FRAME DIAGNOSTIC")
    print(SEP)

    H = history_steps  # present step index

    # ------------------------------------------------------------------
    # SECTION 1: Origin & angle
    # ------------------------------------------------------------------
    print("\n[SECTION 1] ORIGIN & ANGLE")
    print(SEP2)

    origin = data.get("origin", None)
    angle = data.get("angle", None)

    if origin is None:
        print("  ❌ origin: MISSING — cache was not built with first_time=True")
    else:
        o0 = origin[0].tolist()
        om = _absmax(origin)
        print(f"  origin[0]  = {o0}")
        print(f"  origin absmax = {om:.1f}")
        if om > 1000:
            print("  → UTM coordinates — large values expected (e.g. 664000, 3990000)")
        else:
            print("  → Near-zero — origin already relative or unused")

    if angle is None:
        print("  ❌ angle: MISSING")
    else:
        print(f"  angle[0]   = {angle[0].item():.4f} rad")

    # ------------------------------------------------------------------
    # SECTION 2: Agent positions
    # ------------------------------------------------------------------
    print("\n[SECTION 2] AGENT POSITIONS")
    print(SEP2)

    pos = data["agent"]["position"]   # (B, A, T, 2)
    B, A, T, _ = pos.shape
    print(f"  shape: {tuple(pos.shape)}  (B={B}, A={A}, T={T})")

    ego_past = pos[0, 0, :H, :]
    ego_present = pos[0, 0, H - 1, :]
    ego_future = pos[0, 0, H:, :]

    print("\n  EGO (agent 0):")
    print(f"    past    absmax = {_absmax(ego_past):.3f}")
    print(f"    present        = {ego_present.tolist()}")
    print(f"    future  absmax = {_absmax(ego_future):.3f}")
    print(f"    past first 3   = {pos[0, 0, :3, :].tolist()}")
    print(f"    future first 3 = {pos[0, 0, H:H + 3, :].tolist()}")

    if _absmax(ego_present) < 1.0:
        print("    → ego present ≈ (0,0): ego IS at origin ✅")
        print("    → but past/future may still be in UTM if not subtracted")
    else:
        print("    → ego present ≠ (0,0): positions likely in UTM ❌")

    if A > 1:
        other_present = pos[0, 1:min(4, A), H - 1, :]
        other_absmax = _absmax(pos[0, 1:, :, :])
        print(f"\n  OTHER AGENTS (agents 1-{min(3, A - 1)}) at present step:")
        print(f"    {other_present.tolist()}")
        print(f"    other agents absmax = {other_absmax:.3f}")

        if other_absmax > 1000:
            print("    → other agents are in UTM ❌")
        elif other_absmax < 200:
            print("    → other agents look ego-relative ✅")

    # ------------------------------------------------------------------
    # SECTION 3: Agent targets
    # ------------------------------------------------------------------
    print("\n[SECTION 3] AGENT TARGETS")
    print(SEP2)

    if "target" not in data["agent"]:
        print("  ❌ target: MISSING from data['agent']")
    else:
        tgt = data["agent"]["target"]   # (B, A, T_future, 3)
        print(f"  shape: {tuple(tgt.shape)}")

        ego_tgt = tgt[0, 0]
        print("\n  EGO target:")
        print(f"    absmax        = {_absmax(ego_tgt):.3f}")
        print(f"    first 3 steps = {ego_tgt[:3, :2].tolist()}")
        print(f"    last  3 steps = {ego_tgt[-3:, :2].tolist()}")

        if _absmax(ego_tgt) < 50:
            print("    → ego target looks correct (per-agent-relative) ✅")
        else:
            print("    → ego target is too large ❌")

        if A > 1:
            other_tgt = tgt[0, 1:min(4, A)]
            print(f"\n  OTHER AGENTS target (agents 1-{min(3, A - 1)}):")
            print(f"    absmax        = {_absmax(other_tgt):.3f}")
            print(f"    agent1 first 3= {tgt[0, 1, :3, :2].tolist()}")

            if _absmax(other_tgt) < 50:
                print("    → other targets look correct (per-agent-relative) ✅")
            else:
                print("    → other targets are too large ❌")

    # ------------------------------------------------------------------
    # SECTION 4: Map
    # ------------------------------------------------------------------
    print("\n[SECTION 4] MAP")
    print(SEP2)

    if "map" in data:
        pc = data["map"].get("polygon_center", None)
        if pc is not None:
            print(f"  polygon_center shape: {tuple(pc.shape)}")
            print(f"  polygon_center[0,:3] = {pc[0, :3].tolist()}")
            pm = _absmax(pc[..., :2])
            print(f"  polygon_center xy absmax = {pm:.3f}")

            if pm > 1000:
                print("  → map is in UTM ❌")
            elif pm < 200:
                print("  → map looks ego-relative ✅")

        pp = data["map"].get("point_position", None)
        if pp is not None:
            pm2 = _absmax(pp)
            print(f"  point_position absmax = {pm2:.3f}")

    # ------------------------------------------------------------------
    # SECTION 5: Model outputs
    # ------------------------------------------------------------------
    print("\n[SECTION 5] MODEL OUTPUTS")
    print(SEP2)

    traj = res["trajectory"]
    traj_xy = traj[..., :2]
    print(f"  trajectory shape: {tuple(traj.shape)}")
    print(f"  trajectory xy absmax = {_absmax(traj_xy):.3f}")
    print(f"  traj[0,0,0,:3,:2] = {traj_xy[0, 0, 0, :3, :].tolist()}")

    if _absmax(traj_xy) < 100:
        print("  → trajectory is ego-relative ✅")
    else:
        print("  → trajectory is in wrong frame ❌")

    pred = res["prediction"][..., :2]
    print(f"\n  prediction shape: {tuple(res['prediction'].shape)}")
    print(f"  prediction xy absmax = {_absmax(pred):.3f}")
    print(f"  pred[0,0,:3,:2] = {pred[0, :min(3, pred.shape[1]), :3, :].tolist()}")

    if _absmax(pred) < 200:
        print("  → prediction looks ego-relative ✅")
    else:
        print("  → prediction is in wrong frame ❌")

    # ------------------------------------------------------------------
    # SECTION 6: Frame match — traj vs ego target
    # ------------------------------------------------------------------
    print("\n[SECTION 6] FRAME MATCH: trajectory vs ego target")
    print(SEP2)

    if "target" in data["agent"]:
        ego_tgt_xy = data["agent"]["target"][:, 0, :, :2]
        traj_flat = traj[:, 0, :, :, :2]
        T_min = min(ego_tgt_xy.shape[1], traj_flat.shape[2])

        endpoint_dist = torch.norm(
            traj_flat[:, :, T_min - 1, :] - ego_tgt_xy[:, T_min - 1:T_min, :],
            dim=-1,
        )

        min_dist = endpoint_dist.min(dim=-1)[0].mean().item()
        mean_dist = endpoint_dist.mean().item()

        print(f"  best mode endpoint dist (mean over batch): {min_dist:.3f}m")
        print(f"  mean mode endpoint dist:                   {mean_dist:.3f}m")

        if min_dist < 5:
            print("  → trajectory and ego_target are in the SAME frame ✅")
            print("  → metrics should work correctly with data['agent']['target'][:,0]")
        elif min_dist < 20:
            print("  → moderate mismatch — possible scale or offset issue")
        else:
            print("  → trajectory and ego_target are in DIFFERENT frames ❌")
            print("  → _compute_metrics target needs fixing")

    # ------------------------------------------------------------------
    # SECTION 7: Frame match — prediction vs agent targets
    # ------------------------------------------------------------------
    print("\n[SECTION 7] FRAME MATCH: prediction vs agent targets")
    print(SEP2)

    if "target" in data["agent"] and data["agent"]["target"].shape[1] > 1:
        pred_xy = res["prediction"][..., :2]
        per_agent_tgt = data["agent"]["target"][:, 1:, :, :2]
        ego_rel_tgt = data["agent"]["position"][:, 1:, H:, :2]

        T_min = min(pred_xy.shape[2], per_agent_tgt.shape[2])
        vmask = data["agent"]["valid_mask"][:, 1:, H:H + T_min].float()

        diff1 = torch.norm(pred_xy[:, :, :T_min] - per_agent_tgt[:, :, :T_min], dim=-1)
        d1 = (diff1 * vmask).sum() / (vmask.sum() + 1e-6)

        diff2 = torch.norm(pred_xy[:, :, :T_min] - ego_rel_tgt[:, :, :T_min], dim=-1)
        d2 = (diff2 * vmask).sum() / (vmask.sum() + 1e-6)

        print(f"  mean displacement vs data['agent']['target'][:,1:] : {d1.item():.3f}m")
        print(f"  mean displacement vs data['agent']['position'][:,1:,H:] : {d2.item():.3f}m")

        if d1 < d2:
            print("  → prediction matches per-agent-relative target ✅")
            print("  → use data['agent']['target'][:,1:] for PredAvgADE/FDE")
        else:
            print("  → prediction matches ego-relative position ✅")
            print("  → use data['agent']['position'][:,1:,H:] for PredAvgADE/FDE")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  FINAL CONCLUSIONS & RECOMMENDED FIXES")
    print(SEP)

    conclusions = []

    if origin is not None and _absmax(origin) > 1000:
        conclusions.append(
            (
                "DATA FRAME",
                "Cache is in UTM — but position[ego,present]≈(0,0) suggests "
                "PARTIAL normalization. Map and other agents may still be UTM. "
                "Check section 4 map values.",
            )
        )

    if "target" in data["agent"]:
        et = _absmax(data["agent"]["target"][:, 0])
        if et < 50:
            conclusions.append(
                (
                    "EGO TARGET",
                    f"data['agent']['target'][:,0] is correct ({et:.2f}m). "
                    "Use this directly in _compute_metrics.",
                )
            )
        else:
            conclusions.append(
                (
                    "EGO TARGET",
                    f"data['agent']['target'][:,0] is WRONG ({et:.2f}m). "
                    "Needs recomputation.",
                )
            )

    if "target" in data["agent"]:
        ego_tgt_xy = data["agent"]["target"][:, 0, :, :2]
        traj_flat = traj[:, 0, :, :, :2]
        T_min = min(ego_tgt_xy.shape[1], traj_flat.shape[2])

        endpoint_dist = torch.norm(
            traj_flat[:, :, T_min - 1, :] - ego_tgt_xy[:, T_min - 1:T_min, :],
            dim=-1,
        )
        min_dist = endpoint_dist.min(dim=-1)[0].mean().item()

        if min_dist > 20:
            conclusions.append(
                (
                    "_compute_metrics FIX NEEDED",
                    f"trajectory and ego_target endpoint dist={min_dist:.1f}m. "
                    "They are in different frames. Fix _compute_metrics target.",
                )
            )
        else:
            conclusions.append(
                (
                    "_compute_metrics",
                    f"trajectory and ego_target are aligned ({min_dist:.1f}m). "
                    "No fix needed for ego target in _compute_metrics.",
                )
            )

    for label, msg in conclusions:
        print(f"\n  [{label}]")
        print(f"    {msg}")

    print(f"\n{SEP}")
    print("  END DIAGNOSTIC")
    print(f"{SEP}\n")
