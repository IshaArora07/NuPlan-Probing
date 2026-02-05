# EMOE_Planner/hydra_losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ----------------------------
# Config
# ----------------------------

@dataclass
class HydraLossConfig:
    """
    Loss configuration for Hydra-style auxiliary heads.

    Assumes:
      - head outputs are per-mode: [B, Ka]
      - teacher targets are per-mode: [B, Ka]
    """
    # weights
    w_feasibility: float = 0.05
    w_cost: float = 0.10
    w_progress: float = 0.02
    w_comfort: float = 0.02
    w_rank: float = 0.00  # optional listwise ranking distillation from teacher cost

    # loss choices
    use_focal_for_feas: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # regression
    huber_delta: float = 1.0  # SmoothL1 beta
    clamp_targets_cost: Optional[Tuple[float, float]] = (-1e4, 1e4)
    clamp_targets_progress: Optional[Tuple[float, float]] = (-1e3, 1e3)
    clamp_targets_comfort: Optional[Tuple[float, float]] = (0.0, 1e4)

    # uncertainty (if you enabled cost_mean/cost_log_var head outputs)
    enable_cost_uncertainty_nll: bool = False
    clamp_log_var: Tuple[float, float] = (-10.0, 5.0)

    # masking
    # If provided, expects teacher["feasibility"] in [0,1] and modes with 0 can be masked from regressions.
    mask_regression_by_feasibility: bool = True
    feas_mask_threshold: float = 0.5

    # numerical stability
    eps: float = 1e-6


# ----------------------------
# Loss helpers
# ----------------------------

def _safe_mean(x: Tensor, eps: float = 1e-6) -> Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.mean()


def _apply_mask(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return x
    return x[mask]


def _smooth_l1(pred: Tensor, target: Tensor, beta: float) -> Tensor:
    # torch SmoothL1Loss supports beta in newer versions, but implement explicitly for compatibility.
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    return loss


def _bce_with_logits_focal(
    logits: Tensor,
    targets: Tensor,
    alpha: float,
    gamma: float,
) -> Tensor:
    """
    Focal BCE (binary) on logits.
    targets expected in {0,1} or [0,1].
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1.0 - targets) * (1.0 - p)
    focal = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * ((1.0 - pt) ** gamma) * bce
    return focal


def listwise_rank_distill_from_cost(
    pred_cost: Tensor, teacher_cost: Tensor, temperature: float = 1.0
) -> Tensor:
    """
    Listwise ranking distillation:
      p_teacher(k) ∝ exp(-teacher_cost_k / T)
      p_pred(k)    ∝ exp(-pred_cost_k / T)
    Loss = KL(p_teacher || p_pred)

    Shapes: [B, Ka]
    """
    T = max(temperature, 1e-6)
    logp_pred = F.log_softmax(-pred_cost / T, dim=-1)
    p_teacher = F.softmax(-teacher_cost / T, dim=-1).detach()
    # KL: sum p_teacher * (log p_teacher - log p_pred)
    # We can drop the teacher entropy term for optimization equivalence and compute CE:
    loss = -(p_teacher * logp_pred).sum(dim=-1)  # [B]
    return loss


# ----------------------------
# Main loss module
# ----------------------------

class HydraLosses(nn.Module):
    """
    Computes auxiliary losses for Hydra-style heads given:
      - head outputs from HydraPredictionHeads
      - teacher targets from RuleBasedTeachers (Level 1: progress/comfort/basic feasibility/cost)

    Expected keys:
      heads:
        - "feasibility_logits" [B,Ka] (optional)
        - "cost" [B,Ka] OR ("cost_mean","cost_log_var") if uncertainty
        - "progress" [B,Ka] (optional)
        - "comfort" [B,Ka] (optional)
      teacher:
        - "feasibility" [B,Ka]
        - "cost"        [B,Ka]
        - "progress"    [B,Ka]
        - "comfort"     [B,Ka]
    """

    def __init__(self, cfg: HydraLossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        heads: Dict[str, Tensor],
        teacher: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        losses: Dict[str, Tensor] = {}
        total = torch.zeros((), device=next(iter(heads.values())).device, dtype=torch.float32)

        # --- Feasibility ---
        if cfg.w_feasibility > 0.0 and "feasibility_logits" in heads and "feasibility" in teacher:
            logits = heads["feasibility_logits"]
            t = teacher["feasibility"].to(logits.dtype)

            # Ensure shape [B,Ka]
            assert logits.shape == t.shape, f"feasibility shape mismatch: {logits.shape} vs {t.shape}"

            if cfg.use_focal_for_feas:
                l = _bce_with_logits_focal(logits, t, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
            else:
                l = F.binary_cross_entropy_with_logits(logits, t, reduction="none")

            l = _safe_mean(l, cfg.eps)
            losses["loss_feasibility"] = l
            total = total + cfg.w_feasibility * l

        # Feasibility mask for regressions
        feas_mask: Optional[Tensor] = None
        if cfg.mask_regression_by_feasibility and "feasibility" in teacher:
            feas_mask = (teacher["feasibility"] >= cfg.feas_mask_threshold)

        # --- Cost regression / NLL ---
        if cfg.w_cost > 0.0 and "cost" in teacher:
            t_cost = teacher["cost"]
            if cfg.clamp_targets_cost is not None:
                t_cost = torch.clamp(t_cost, cfg.clamp_targets_cost[0], cfg.clamp_targets_cost[1])
            t_cost = t_cost.to(torch.float32)

            if cfg.enable_cost_uncertainty_nll and ("cost_mean" in heads and "cost_log_var" in heads):
                mu = heads["cost_mean"].to(torch.float32)
                log_var = heads["cost_log_var"].to(torch.float32)
                assert mu.shape == t_cost.shape, f"cost_mean mismatch: {mu.shape} vs {t_cost.shape}"
                assert log_var.shape == t_cost.shape, f"cost_log_var mismatch: {log_var.shape} vs {t_cost.shape}"

                log_var = torch.clamp(log_var, cfg.clamp_log_var[0], cfg.clamp_log_var[1])
                var = torch.exp(log_var)

                # Gaussian NLL up to constant: 0.5*(log_var + (x-mu)^2/var)
                nll = 0.5 * (log_var + ((t_cost - mu) ** 2) / (var + cfg.eps))
                nll = _apply_mask(nll, feas_mask)
                l = _safe_mean(nll, cfg.eps)
                losses["loss_cost_nll"] = l
                total = total + cfg.w_cost * l
            else:
                if "cost" not in heads:
                    # If model didn't output cost head, skip cleanly.
                    pass
                else:
                    p_cost = heads["cost"].to(torch.float32)
                    assert p_cost.shape == t_cost.shape, f"cost mismatch: {p_cost.shape} vs {t_cost.shape}"

                    reg = _smooth_l1(p_cost, t_cost, beta=cfg.huber_delta)
                    reg = _apply_mask(reg, feas_mask)
                    l = _safe_mean(reg, cfg.eps)
                    losses["loss_cost"] = l
                    total = total + cfg.w_cost * l

        # --- Progress regression ---
        if cfg.w_progress > 0.0 and "progress" in heads and "progress" in teacher:
            p = heads["progress"].to(torch.float32)
            t = teacher["progress"].to(torch.float32)
            if cfg.clamp_targets_progress is not None:
                t = torch.clamp(t, cfg.clamp_targets_progress[0], cfg.clamp_targets_progress[1])
            assert p.shape == t.shape, f"progress mismatch: {p.shape} vs {t.shape}"

            reg = _smooth_l1(p, t, beta=cfg.huber_delta)
            reg = _apply_mask(reg, feas_mask)
            l = _safe_mean(reg, cfg.eps)
            losses["loss_progress"] = l
            total = total + cfg.w_progress * l

        # --- Comfort regression ---
        if cfg.w_comfort > 0.0 and "comfort" in heads and "comfort" in teacher:
            p = heads["comfort"].to(torch.float32)
            t = teacher["comfort"].to(torch.float32)
            if cfg.clamp_targets_comfort is not None:
                t = torch.clamp(t, cfg.clamp_targets_comfort[0], cfg.clamp_targets_comfort[1])
            assert p.shape == t.shape, f"comfort mismatch: {p.shape} vs {t.shape}"

            reg = _smooth_l1(p, t, beta=cfg.huber_delta)
            reg = _apply_mask(reg, feas_mask)
            l = _safe_mean(reg, cfg.eps)
            losses["loss_comfort"] = l
            total = total + cfg.w_comfort * l

        # --- Optional ranking distillation (listwise) ---
        # Requires model cost head AND teacher cost.
        if cfg.w_rank > 0.0 and "cost" in teacher and ("cost" in heads):
            p_cost = heads["cost"].to(torch.float32)
            t_cost = teacher["cost"].to(torch.float32)
            # For ranking, do NOT mask (infeasible costs should be high; but if your teacher doesn't encode that, mask)
            rank_loss = listwise_rank_distill_from_cost(p_cost, t_cost, temperature=1.0)
            rank_loss = _safe_mean(rank_loss, cfg.eps)
            losses["loss_rank"] = rank_loss
            total = total + cfg.w_rank * rank_loss

        losses["loss_hydra_total"] = total
        return losses
