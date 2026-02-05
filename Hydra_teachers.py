# EMOE_Planner/rule_based_teachers.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

Tensor = torch.Tensor


# ----------------------------
# Config
# ----------------------------

@dataclass
class RuleTeacherConfig:
    """
    Rule-based teacher targets computed from predicted trajectories.

    Assumes planner outputs:
      - traj:   [B, Ka, T, 6]
      - scores: [B, Ka]  (optional, not used by default teachers)

    By convention, traj[..., 0:2] are (x, y) in some metric frame (global or local).
    If your 6D ordering differs, set xy_indices accordingly.
    """
    # Which indices in the 6D state correspond to x,y
    xy_indices: Tuple[int, int] = (0, 1)

    # Time step (seconds). If unknown, set to 1.0 and treat comfort as relative.
    dt: float = 0.5

    # Feasibility rules
    max_speed: Optional[float] = None          # m/s, if you have v in state and want to enforce
    speed_index: Optional[int] = None          # index of speed in 6D state, if present
    max_step_distance: Optional[float] = 15.0  # m per step; guards numerical explosions
    nan_is_infeasible: bool = True

    # Cost weights (teacher cost = weighted sum)
    w_collision: float = 1000.0
    w_offroad: float = 300.0
    w_comfort: float = 1.0
    w_progress: float = -5.0  # negative means "more progress => lower cost"

    # Clamps to keep targets numerically stable
    clamp_cost: Tuple[float, float] = (-1e4, 1e4)
    clamp_progress: Tuple[float, float] = (-1e3, 1e3)
    clamp_comfort: Tuple[float, float] = (0.0, 1e4)

    # Whether to output soft feasibility in [0,1] (else hard {0,1})
    soft_feasibility: bool = True
    # softness scale for soft feasibility (higher => more forgiving)
    feasibility_softness: float = 10.0


# ----------------------------
# Optional external evaluators
# ----------------------------

# These are hooks you can provide from your nuPlan / map / collision code.
# They should be vectorized and return tensors aligned with [B, Ka].
CollisionEvaluator = Callable[[Tensor, Dict[str, Any]], Tensor]
OffroadEvaluator = Callable[[Tensor, Dict[str, Any]], Tensor]


# ----------------------------
# Utilities
# ----------------------------

def _xy(traj: Tensor, xy_idx: Tuple[int, int]) -> Tensor:
    """Extract XY: [B, Ka, T, 2]."""
    x_i, y_i = xy_idx
    return traj[..., [x_i, y_i]]


def _finite_mask(x: Tensor) -> Tensor:
    """True where all elements are finite (no NaN/Inf) along the last dim(s)."""
    return torch.isfinite(x).all(dim=-1)


def _pairwise_deltas(xy: Tensor) -> Tensor:
    """xy: [B, Ka, T, 2] -> dxy: [B, Ka, T-1, 2]."""
    return xy[..., 1:, :] - xy[..., :-1, :]


def _speed_from_xy(xy: Tensor, dt: float) -> Tensor:
    """xy: [B, Ka, T, 2] -> speed: [B, Ka, T-1]."""
    dxy = _pairwise_deltas(xy)
    dist = torch.linalg.norm(dxy, dim=-1)  # [B, Ka, T-1]
    return dist / max(dt, 1e-6)


def _jerk_proxy_from_xy(xy: Tensor, dt: float) -> Tensor:
    """
    Comfort proxy: mean squared jerk magnitude derived from xy.

    Steps:
      v_t = (p_{t+1}-p_t)/dt
      a_t = (v_{t+1}-v_t)/dt
      j_t = (a_{t+1}-a_t)/dt
    Output:
      comfort: [B, Ka]  (higher => worse)
    """
    v = _pairwise_deltas(xy) / max(dt, 1e-6)                 # [B, Ka, T-1, 2]
    if v.shape[-2] < 3:
        return torch.zeros(xy.shape[0], xy.shape[1], device=xy.device, dtype=xy.dtype)

    a = (v[..., 1:, :] - v[..., :-1, :]) / max(dt, 1e-6)     # [B, Ka, T-2, 2]
    if a.shape[-2] < 2:
        return torch.zeros(xy.shape[0], xy.shape[1], device=xy.device, dtype=xy.dtype)

    j = (a[..., 1:, :] - a[..., :-1, :]) / max(dt, 1e-6)     # [B, Ka, T-3, 2]
    j2 = (j ** 2).sum(dim=-1)                                 # [B, Ka, T-3]
    return j2.mean(dim=-1)                                    # [B, Ka]


def _progress_from_xy(xy: Tensor) -> Tensor:
    """
    Simple progress teacher: net displacement magnitude over horizon.

    progress = ||p_T - p_0||, output [B, Ka].
    You can replace this with route-aligned progress later.
    """
    p0 = xy[..., 0, :]      # [B, Ka, 2]
    pT = xy[..., -1, :]     # [B, Ka, 2]
    return torch.linalg.norm(pT - p0, dim=-1)  # [B, Ka]


def _basic_feasibility(
    traj: Tensor,
    xy: Tensor,
    cfg: RuleTeacherConfig,
) -> Tensor:
    """
    Basic feasibility rules that require no map:
      - finite check
      - step-distance explosion check
      - optional max_speed check (either from xy or from speed channel)
    Returns:
      infeasible_mask: [B, Ka] bool
    """
    B, Ka, T, _ = traj.shape

    # NaN/Inf check across all state dims and time
    finite = torch.isfinite(traj).all(dim=-1).all(dim=-1)  # [B, Ka]
    infeasible = ~finite if cfg.nan_is_infeasible else torch.zeros_like(finite, dtype=torch.bool)

    # Step distance explosion (from xy)
    dxy = _pairwise_deltas(xy)  # [B, Ka, T-1, 2]
    step_dist = torch.linalg.norm(dxy, dim=-1)             # [B, Ka, T-1]
    if cfg.max_step_distance is not None:
        infeasible = infeasible | (step_dist.max(dim=-1).values > cfg.max_step_distance)

    # Optional speed constraint
    if cfg.max_speed is not None:
        if cfg.speed_index is not None:
            v = traj[..., cfg.speed_index]  # [B, Ka, T]
            infeasible = infeasible | (v.max(dim=-1).values > cfg.max_speed)
        else:
            v_xy = _speed_from_xy(xy, cfg.dt)  # [B, Ka, T-1]
            infeasible = infeasible | (v_xy.max(dim=-1).values > cfg.max_speed)

    return infeasible


def _soften_feasibility(hard_feasible: Tensor, softness: float) -> Tensor:
    """
    Convert hard feasibility {0,1} to soft in [0,1] with a fixed softness.
    This is intentionally simple; if you later have margin distances, use those instead.
    """
    # hard_feasible: [B, Ka] in {0,1}
    # soft: 0.0 for infeasible, ~1.0 for feasible
    # softness only used to keep interface consistent; here it’s binary-soft.
    return hard_feasible.to(torch.float32)


# ----------------------------
# Main teacher
# ----------------------------

class RuleBasedTeachers:
    """
    Computes rule-based teacher targets for Hydra heads.

    You can optionally supply:
      - collision_evaluator(traj, context) -> [B, Ka] collision_cost (>=0)
      - offroad_evaluator(traj, context)   -> [B, Ka] offroad_cost (>=0)

    If not supplied, collision/offroad costs default to 0 and feasibility is based on basic rules only.
    """

    def __init__(
        self,
        cfg: RuleTeacherConfig,
        collision_evaluator: Optional[CollisionEvaluator] = None,
        offroad_evaluator: Optional[OffroadEvaluator] = None,
    ) -> None:
        self.cfg = cfg
        self.collision_evaluator = collision_evaluator
        self.offroad_evaluator = offroad_evaluator

    @torch.no_grad()
    def __call__(
        self,
        traj: Tensor,                      # [B, Ka, T, 6]
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        """
        Returns teacher targets aligned with modes:

          feasibility: [B, Ka] in {0,1} or [0,1]
          collision_cost: [B, Ka]
          offroad_cost:   [B, Ka]
          comfort:        [B, Ka]
          progress:       [B, Ka]
          cost:           [B, Ka]  (weighted sum, clamped)

        Notes:
          - "cost" is a scalar that you can regress or use for ranking distillation.
          - If you later want route-progress, pass a richer context and replace _progress_from_xy().
        """
        if context is None:
            context = {}

        assert traj.dim() == 4, f"Expected traj [B,Ka,T,6], got {tuple(traj.shape)}"
        B, Ka, T, S = traj.shape

        xy = _xy(traj, self.cfg.xy_indices)  # [B, Ka, T, 2]

        # Base components
        progress = _progress_from_xy(xy)                 # [B, Ka]
        comfort = _jerk_proxy_from_xy(xy, self.cfg.dt)   # [B, Ka]

        # Optional map/collision evaluators (must return non-negative costs)
        if self.collision_evaluator is not None:
            collision_cost = self.collision_evaluator(traj, context).to(traj.dtype)
        else:
            collision_cost = torch.zeros(B, Ka, device=traj.device, dtype=traj.dtype)

        if self.offroad_evaluator is not None:
            offroad_cost = self.offroad_evaluator(traj, context).to(traj.dtype)
        else:
            offroad_cost = torch.zeros(B, Ka, device=traj.device, dtype=traj.dtype)

        # Basic feasibility (no-map) plus any evaluator-based feasibility gates if desired
        infeasible = _basic_feasibility(traj, xy, self.cfg)  # [B, Ka] bool

        # If evaluator costs exist, treat >0 as infeasible only if you explicitly want that behavior.
        # By default, we do NOT hard-gate on these costs; they contribute to cost.
        hard_feasible = (~infeasible).to(torch.float32)  # [B, Ka]

        if self.cfg.soft_feasibility:
            feasibility = _soften_feasibility(hard_feasible, self.cfg.feasibility_softness)
        else:
            feasibility = hard_feasible

        # Clamp base components for stability
        progress = torch.clamp(progress, self.cfg.clamp_progress[0], self.cfg.clamp_progress[1])
        comfort = torch.clamp(comfort, self.cfg.clamp_comfort[0], self.cfg.clamp_comfort[1])

        # Weighted teacher cost (lower is better)
        cost = (
            self.cfg.w_collision * collision_cost
            + self.cfg.w_offroad * offroad_cost
            + self.cfg.w_comfort * comfort
            + self.cfg.w_progress * progress
        )
        cost = torch.clamp(cost, self.cfg.clamp_cost[0], self.cfg.clamp_cost[1])

        return {
            "feasibility": feasibility,           # [B, Ka]
            "collision_cost": collision_cost,     # [B, Ka]
            "offroad_cost": offroad_cost,         # [B, Ka]
            "comfort": comfort,                   # [B, Ka]
            "progress": progress,                 # [B, Ka]
            "cost": cost,                         # [B, Ka]
        }


# ----------------------------
# Optional: ranking labels helper
# ----------------------------

@torch.no_grad()
def teacher_ranking_from_cost(cost: Tensor) -> Tensor:
    """
    Convert cost [B, Ka] into a soft ranking distribution over modes:
      p(k) ∝ exp(-cost_k)
    Useful for listwise distillation.
    """
    assert cost.dim() == 2, f"Expected [B,Ka], got {tuple(cost.shape)}"
    return torch.softmax(-cost, dim=-1)
