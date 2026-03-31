#!/usr/bin/env python3
"""
src/utils/diagnose_emoe.py

Full diagnostic for the EMoE planner. Checks coordinate frames,
loss computation, metric correctness, router behaviour, and model
output sanity in a single forward pass.

Usage in pluto_trainer.py _step():

from src.utils.diagnose_emoe import diagnose_emoe

def _step(self, batch, prefix):
    features, targets, scenarios = batch
    data = features["feature"].data

    if self.global_step == 0 and prefix == "train":
        res = self.forward(data)
        diagnose_emoe(self, data, res)
        raise SystemExit("EMoE diagnostic complete — check output, then remove this block")

    res = self.forward(data)
    ...
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SEP = "=" * 65
SEP2 = "-" * 55
OK = "✅"
ERR = "❌"
WARN = "⚠️"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _absmax(t: torch.Tensor) -> float:
    f = t.float()
    finite = f[torch.isfinite(f)]
    return finite.abs().max().item() if finite.numel() > 0 else float("nan")


def _has_nan(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()


def _check(label: str, condition: bool, good_msg: str, bad_msg: str) -> bool:
    icon = OK if condition else ERR
    msg = good_msg if condition else bad_msg
    print(f"  {icon} {label}: {msg}")
    return condition


def _section(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ──────────────────────────────────────────────────────────────
# Main diagnostic
# ──────────────────────────────────────────────────────────────
def diagnose_emoe(trainer, data: dict, res: dict) -> None:
    H = trainer.history_steps
    S = trainer.num_scene_types
    Ka = trainer.num_modes

    issues = []
    warnings = []

    print(f"\n{SEP}")
    print("  EMoE FULL DIAGNOSTIC")
    print(SEP)

    # ══════════════════════════════════════════════════════════
    # 1. INPUT DATA — coordinate frames
    # ══════════════════════════════════════════════════════════
    _section("1. INPUT DATA — COORDINATE FRAMES")

    pos = data["agent"]["position"]  # (B, A, T, 2)
    B, A, T_total, _ = pos.shape
    print(f"  batch size B={B}, agents A={A}, timesteps T={T_total}, history_steps H={H}")

    # origin / angle
    origin_ok = "origin" in data and "angle" in data
    _check(
        "origin/angle stored",
        origin_ok,
        f"origin absmax={_absmax(data['origin']):.1f}",
        "MISSING — cache not built with first_time=True",
    )
    if not origin_ok:
        issues.append(("CACHE", "origin/angle missing — rebuild cache with first_time=True"))

    # ego at present step
    ego_present = pos[0, 0, H - 1, :]
    ego_at_origin = ego_present.abs().max().item() < 1.0
    _check(
        "ego at origin at present step",
        ego_at_origin,
        f"ego present={ego_present.tolist()}",
        f"ego present={ego_present.tolist()} — not at (0,0), unexpected",
    )
    if not ego_at_origin:
        issues.append(
            ("FRAME", "ego position at present step is not (0,0) — normalization may have failed")
        )

    # other agents plausible range
    if A > 1:
        other_max = _absmax(pos[:, 1:])
        other_ok = other_max < 300
        _check(
            "other agents in plausible range",
            other_ok,
            f"absmax={other_max:.1f}m",
            f"absmax={other_max:.1f}m — too large, may be UTM",
        )
        if not other_ok:
            issues.append(
                (
                    "FRAME",
                    f"other agents absmax={other_max:.1f}m — likely UTM, normalize_batch needed",
                )
            )

    # map
    if "map" in data and "polygon_center" in data["map"]:
        map_max = _absmax(data["map"]["polygon_center"][..., :2])
        map_ok = map_max < 300
        _check(
            "map in plausible range",
            map_ok,
            f"polygon_center absmax={map_max:.1f}m",
            f"polygon_center absmax={map_max:.1f}m — too large, may be UTM",
        )
        if not map_ok:
            issues.append(("FRAME", f"map absmax={map_max:.1f}m — likely UTM"))

    # ego target
    if "target" in data["agent"]:
        tgt = data["agent"]["target"]  # (B, A, T_future, 3)
        ego_tgt_max = _absmax(tgt[:, 0])
        ego_tgt_ok = ego_tgt_max < 80
        _check(
            "ego target in plausible range",
            ego_tgt_ok,
            f"absmax={ego_tgt_max:.3f}m",
            f"absmax={ego_tgt_max:.3f}m — too large",
        )
        if not ego_tgt_ok:
            issues.append(("TARGET", f"ego target absmax={ego_tgt_max:.1f}m — wrong frame"))

        if A > 1:
            other_tgt_max = _absmax(tgt[:, 1:])
            other_tgt_ok = other_tgt_max < 80
            _check(
                "other agent targets in plausible range",
                other_tgt_ok,
                f"absmax={other_tgt_max:.3f}m",
                f"absmax={other_tgt_max:.3f}m — too large",
            )
            if not other_tgt_ok:
                issues.append(
                    ("TARGET", f"other agent targets absmax={other_tgt_max:.1f}m — wrong frame")
                )
    else:
        issues.append(("TARGET", "data['agent']['target'] missing entirely"))

    # cost maps
    if "cost_maps" in data:
        cm_max = _absmax(data["cost_maps"])
        _check(
            "cost_maps present and finite",
            cm_max < 1e6,
            f"absmax={cm_max:.3f}",
            f"absmax={cm_max:.3f} — may contain inf/nan",
        )
    else:
        warnings.append(("COLLISION", "cost_maps missing — collision loss will be skipped"))

    # EMoE labels
    if "emoe" in data and "emoe_class_id" in data["emoe"]:
        labels = data["emoe"]["emoe_class_id"]
        lmin, lmax = labels.min().item(), labels.max().item()
        labels_ok = lmin >= 0 and lmax < S
        _check(
            "emoe_class_id range",
            labels_ok,
            f"range=[{lmin},{lmax}], S={S}",
            f"range=[{lmin},{lmax}] out of [0,{S - 1}]",
        )
        if not labels_ok:
            issues.append(("ROUTER", f"emoe_class_id out of range [0,{S - 1}]"))

        unique_labels = labels.unique()
        all_classes_present = unique_labels.numel() == S
        _check(
            "all scene classes present in batch",
            all_classes_present,
            f"unique={unique_labels.tolist()}",
            f"only {unique_labels.tolist()} present — some classes missing from batch",
        )
        if not all_classes_present:
            warnings.append(
                ("ROUTER", f"not all {S} scene classes in this batch — ok if batch is small")
            )
    else:
        issues.append(("ROUTER", "emoe_class_id missing from data['emoe']"))

    # ══════════════════════════════════════════════════════════
    # 2. MODEL OUTPUTS — shapes and sanity
    # ══════════════════════════════════════════════════════════
    _section("2. MODEL OUTPUTS — SHAPES & SANITY")

    traj = res["trajectory"]  # (B, R, Ka, T, 6)
    print(f"  trajectory shape: {tuple(traj.shape)}")
    traj_ok = traj.dim() == 5 and traj.shape[0] == B
    _check(
        "trajectory shape correct",
        traj_ok,
        f"(B={traj.shape[0]}, R={traj.shape[1]}, Ka={traj.shape[2]}, T={traj.shape[3]}, 6)",
        f"unexpected shape {tuple(traj.shape)}",
    )
    _check(
        "trajectory finite",
        not _has_nan(traj),
        "no nan/inf",
        "CONTAINS NAN/INF — training will diverge",
    )
    if _has_nan(traj):
        issues.append(("MODEL", "trajectory contains nan/inf"))

    traj_max = _absmax(traj[..., :2])
    _check(
        "trajectory xy in plausible range",
        traj_max < 200,
        f"absmax={traj_max:.3f}m",
        f"absmax={traj_max:.3f}m — too large",
    )

    prob = res["probability"]  # (B, R, Ka)
    print(f"  probability shape: {tuple(prob.shape)}")
    _check(
        "probability finite",
        not _has_nan(prob),
        "no nan/inf",
        "CONTAINS NAN/INF",
    )
    prob_flat = prob.reshape(B, -1)
    print(f"  probability range: [{prob_flat.min().item():.3f}, {prob_flat.max().item():.3f}]")
    print("  NOTE: raw logits expected (no softmax) — large values ok")

    pred = res["prediction"]  # (B, A-1, T, 2 or more)
    print(f"  prediction shape: {tuple(pred.shape)}")
    _check(
        "prediction finite",
        not _has_nan(pred),
        "no nan/inf",
        "CONTAINS NAN/INF",
    )
    pred_max = _absmax(pred[..., :2])
    _check(
        "prediction xy in plausible range",
        pred_max < 300,
        f"absmax={pred_max:.3f}m",
        f"absmax={pred_max:.3f}m — too large",
    )

    if "router_logits" in res:
        rl = res["router_logits"]
        print(f"  router_logits shape: {tuple(rl.shape)}")
        _check(
            "router_logits shape",
            rl.shape == (B, S),
            f"(B={B}, S={S})",
            f"expected ({B},{S}) got {tuple(rl.shape)}",
        )
        _check(
            "router_logits finite",
            not _has_nan(rl),
            "no nan/inf",
            "CONTAINS NAN/INF",
        )
    else:
        issues.append(("ROUTER", "router_logits missing from res"))

    if "router_idx" in res:
        ri = res["router_idx"]
        ri_ok = ri.min().item() >= 0 and ri.max().item() < S
        _check(
            "router_idx in range",
            ri_ok,
            f"range=[{ri.min().item()},{ri.max().item()}]",
            f"out of [0,{S - 1}]",
        )
    else:
        issues.append(("ROUTER", "router_idx missing from res"))

    # ══════════════════════════════════════════════════════════
    # 3. FRAME ALIGNMENT
    # ══════════════════════════════════════════════════════════
    _section("3. FRAME ALIGNMENT")

    if "target" in data["agent"]:
        ego_tgt_xy = data["agent"]["target"][:, 0, :, :2]
        traj_xy = traj[:, 0, :, :, :2]
        T_min = min(ego_tgt_xy.shape[1], traj_xy.shape[2])

        endpoint_dist = torch.norm(
            traj_xy[:, :, T_min - 1, :] - ego_tgt_xy[:, T_min - 1 : T_min, :],
            dim=-1,
        )
        min_ep = endpoint_dist.min(dim=-1)[0].mean().item()
        mean_ep = endpoint_dist.mean().item()

        print(f"  traj vs ego_target — best mode endpoint dist: {min_ep:.3f}m")
        print(f"  traj vs ego_target — mean mode endpoint dist: {mean_ep:.3f}m")

        aligned = min_ep < 30
        _check(
            "traj and ego_target in same frame",
            aligned,
            f"best endpoint dist={min_ep:.2f}m (untrained model)",
            f"best endpoint dist={min_ep:.2f}m — WRONG FRAME",
        )
        if not aligned:
            issues.append(("FRAME", f"trajectory and ego_target misaligned by {min_ep:.1f}m"))

        if pred.shape[1] > 0 and data["agent"]["target"].shape[1] > 1:
            per_agent_tgt = data["agent"]["target"][:, 1:, :, :2]
            ego_rel_pos = data["agent"]["position"][:, 1:, H:, :2]
            T2 = min(pred.shape[2], per_agent_tgt.shape[2])
            vmask = data["agent"]["valid_mask"][:, 1:, H : H + T2].float()

            d_per_agent = (
                torch.norm(pred[:, :, :T2, :2] - per_agent_tgt[:, :, :T2], dim=-1) * vmask
            ).sum() / (vmask.sum() + 1e-6)

            d_ego_rel = (
                torch.norm(pred[:, :, :T2, :2] - ego_rel_pos[:, :, :T2], dim=-1) * vmask
            ).sum() / (vmask.sum() + 1e-6)

            print(f"\n  pred vs data['agent']['target'][:,1:]      : {d_per_agent.item():.3f}m")
            print(f"  pred vs data['agent']['position'][:,1:,H:] : {d_ego_rel.item():.3f}m")

            if d_per_agent < d_ego_rel:
                print(f"  {OK} prediction matches per-agent-relative target")
                print("     → use data['agent']['target'][:,1:] for PredAvgADE/FDE")
            else:
                print(f"  {OK} prediction matches ego-relative position")
                print("     → use data['agent']['position'][:,1:,H:] for PredAvgADE/FDE")
                warnings.append(
                    (
                        "METRICS",
                        "PredAvgADE/FDE target should be position[:,1:,H:] not target[:,1:]",
                    )
                )

    # ══════════════════════════════════════════════════════════
    # 4. LOSS COMPUTATION SANITY
    # ══════════════════════════════════════════════════════════
    _section("4. LOSS COMPUTATION SANITY")

    if "target" in data["agent"]:
        T_future = data["agent"]["target"].shape[2]
        ego_tgt_6d = data["agent"]["target"][:, 0]
        best_traj = traj[:, 0, 0]
        T2 = min(T_future, best_traj.shape[1])

        reg_check = F.smooth_l1_loss(
            best_traj[:, :T2, :2],
            ego_tgt_6d[:, :T2, :2],
            reduction="mean",
        )

        print(f"  reg_loss (mode0 vs ego_target, xy only): {reg_check.item():.4f}")
        reg_ok = reg_check.item() < 500
        _check(
            "reg_loss is finite and plausible",
            reg_ok,
            f"{reg_check.item():.4f}",
            f"{reg_check.item():.4f} — too large, check frame alignment",
        )
        if not reg_ok:
            issues.append(("LOSS", f"reg_loss={reg_check.item():.1f} — likely frame mismatch"))

    if "router_logits" in res and "emoe" in data:
        labels = data["emoe"]["emoe_class_id"][:B].long()
        router_ce = F.cross_entropy(res["router_logits"][:B], labels)
        max_ce = math.log(S) * 3
        _check(
            "router CE loss plausible",
            router_ce.item() < max_ce,
            f"CE={router_ce.item():.4f} (random baseline={math.log(S):.4f})",
            f"CE={router_ce.item():.4f} — exceeds 3x random baseline, check labels",
        )

    # ══════════════════════════════════════════════════════════
    # 5. ROUTER DIAGNOSTICS
    # ══════════════════════════════════════════════════════════
    _section("5. ROUTER DIAGNOSTICS")

    if "router_logits" in res and "emoe" in data and "emoe_class_id" in data["emoe"]:
        logits = res["router_logits"][:B]
        labels = data["emoe"]["emoe_class_id"][:B].long()
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        acc = (preds == labels).float().mean().item()
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
        max_ent = math.log(S)

        print(f"  router accuracy       : {acc:.3f}  (random={1 / S:.3f})")
        print(f"  router entropy        : {ent:.3f}  (max={max_ent:.3f})")
        print(f"  predicted labels      : {preds.tolist()}")
        print(f"  true labels           : {labels.tolist()}")

        usage = torch.bincount(preds, minlength=S).float() / B
        for i in range(S):
            state = "active" if usage[i].item() > 0 else "DEAD — never selected"
            print(f"  expert {i} usage: {usage[i].item():.3f}  ({state})")

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(SEP)

    if not issues and not warnings:
        print(f"  {OK} Everything looks correct. Safe to run full training.")
    else:
        if issues:
            print(f"\n  {ERR} ISSUES (must fix before training):")
            for label, msg in issues:
                print(f"    [{label}] {msg}")

        if warnings:
            print(f"\n  {WARN} WARNINGS (investigate but may be ok):")
            for label, msg in warnings:
                print(f"    [{label}] {msg}")

    print(f"\n{SEP}")
    print("  END DIAGNOSTIC")
    print(f"{SEP}\n")
