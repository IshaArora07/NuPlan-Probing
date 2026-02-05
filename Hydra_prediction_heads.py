# EMOE_Planner/hydra_prediction_heads.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class HydraHeadsConfig:
    """
    Configuration for Hydra-style auxiliary prediction heads.

    The heads are intended to operate on per-mode latent features Z:
        Z: [B, K, D]  (preferred)
    but also support:
        Z: [B, D]     (single mode; treated as K=1)

    Heads:
      - feasibility: binary validity / constraint satisfaction per mode
      - cost:        scalar cost per mode (regression)
      - progress:    scalar progress metric per mode (regression)
      - comfort:     scalar comfort metric per mode (regression) [optional]
      - uncertainty: predicts (mean, log_var) for cost [optional]
    """
    in_dim: int

    # shared trunk
    trunk_hidden_dim: int = 256
    trunk_depth: int = 2
    trunk_dropout: float = 0.0
    use_layernorm: bool = True

    # enable/disable heads
    enable_feasibility: bool = True
    enable_cost: bool = True
    enable_progress: bool = True
    enable_comfort: bool = False
    enable_uncertainty: bool = False  # if True, outputs cost_mean and cost_log_var

    # per-head hidden dims (small by default)
    head_hidden_dim: int = 128
    head_depth: int = 2
    head_dropout: float = 0.0

    # optional conditioning
    router_probs_dim: int = 0  # if >0, concatenates router_probs to Z at input


class MLP(nn.Module):
    """Small MLP block used for trunk and heads."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        assert depth >= 1, "depth must be >= 1"

        layers = []
        d = in_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, out_dim))
        if activate_last:
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HydraPredictionHeads(nn.Module):
    """
    Hydra-style auxiliary heads operating on per-mode latent features.

    Input:
      - z: [B, K, D] or [B, D]
      - router_probs (optional): [B, K, R] or [B, R] if cfg.router_probs_dim > 0

    Output: dict with keys (depending on cfg):
      - "feasibility_logits": [B, K]    (use sigmoid in loss/metrics)
      - "cost":              [B, K]
      - "progress":          [B, K]
      - "comfort":           [B, K]
      - "cost_mean":         [B, K]    (if uncertainty enabled)
      - "cost_log_var":      [B, K]    (if uncertainty enabled)
      - "trunk_features":    [B, K, H] (optional debugging/analysis)
    """
    def __init__(self, cfg: HydraHeadsConfig) -> None:
        super().__init__()
        self.cfg = cfg

        trunk_in_dim = cfg.in_dim + int(cfg.router_probs_dim)
        self.trunk = MLP(
            in_dim=trunk_in_dim,
            hidden_dim=cfg.trunk_hidden_dim,
            out_dim=cfg.trunk_hidden_dim,
            depth=cfg.trunk_depth,
            dropout=cfg.trunk_dropout,
            use_layernorm=cfg.use_layernorm,
            activate_last=True,
        )

        # Heads are small MLPs from trunk_hidden_dim -> 1 per mode.
        def make_scalar_head() -> nn.Module:
            return MLP(
                in_dim=cfg.trunk_hidden_dim,
                hidden_dim=cfg.head_hidden_dim,
                out_dim=1,
                depth=cfg.head_depth,
                dropout=cfg.head_dropout,
                use_layernorm=cfg.use_layernorm,
                activate_last=False,
            )

        self.feas_head = make_scalar_head() if cfg.enable_feasibility else None

        # Cost head: either deterministic scalar, or (mean, log_var)
        if cfg.enable_cost:
            if cfg.enable_uncertainty:
                self.cost_mean_head = make_scalar_head()
                self.cost_logvar_head = make_scalar_head()
                self.cost_head = None
            else:
                self.cost_head = make_scalar_head()
                self.cost_mean_head = None
                self.cost_logvar_head = None
        else:
            self.cost_head = None
            self.cost_mean_head = None
            self.cost_logvar_head = None

        self.prog_head = make_scalar_head() if cfg.enable_progress else None
        self.comfort_head = make_scalar_head() if cfg.enable_comfort else None

    @staticmethod
    def _ensure_bkd(x: Tensor) -> Tensor:
        """Convert [B, D] -> [B, 1, D]; keep [B, K, D] unchanged."""
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected input with dim 2 or 3, got shape {tuple(x.shape)}")

    def forward(
        self,
        z: Tensor,
        router_probs: Optional[Tensor] = None,
        return_trunk_features: bool = False,
    ) -> Dict[str, Tensor]:
        z_bkd = self._ensure_bkd(z)  # [B, K, D]
        B, K, D = z_bkd.shape

        if self.cfg.router_probs_dim > 0:
            if router_probs is None:
                raise ValueError("cfg.router_probs_dim > 0 but router_probs is None")
            rp = self._ensure_bkd(router_probs)  # [B, K, R] or [B, 1, R]
            if rp.shape[0] != B:
                raise ValueError(f"router_probs batch mismatch: {rp.shape[0]} vs {B}")
            if rp.shape[1] not in (1, K):
                raise ValueError(f"router_probs K mismatch: {rp.shape[1]} vs {K}")
            if rp.shape[2] != self.cfg.router_probs_dim:
                raise ValueError(
                    f"router_probs_dim mismatch: expected {self.cfg.router_probs_dim}, got {rp.shape[2]}"
                )
            if rp.shape[1] == 1 and K > 1:
                rp = rp.expand(B, K, rp.shape[2])
            x = torch.cat([z_bkd, rp], dim=-1)
        else:
            x = z_bkd

        # Flatten modes for MLP: [B*K, ...]
        x_flat = x.reshape(B * K, -1)
        trunk_flat = self.trunk(x_flat)  # [B*K, H]
        trunk = trunk_flat.reshape(B, K, -1)  # [B, K, H]

        out: Dict[str, Tensor] = {}

        def head_forward(head: nn.Module, name: str) -> None:
            y = head(trunk_flat).reshape(B, K)  # scalar per mode
            out[name] = y

        if self.feas_head is not None:
            head_forward(self.feas_head, "feasibility_logits")

        if self.cfg.enable_cost:
            if self.cfg.enable_uncertainty:
                mean = self.cost_mean_head(trunk_flat).reshape(B, K)
                log_var = self.cost_logvar_head(trunk_flat).reshape(B, K)
                # Optional stabilization: clamp log_var to avoid extreme exp() in NLL
                log_var = torch.clamp(log_var, min=-10.0, max=5.0)
                out["cost_mean"] = mean
                out["cost_log_var"] = log_var
            else:
                head_forward(self.cost_head, "cost")

        if self.prog_head is not None:
            head_forward(self.prog_head, "progress")

        if self.comfort_head is not None:
            head_forward(self.comfort_head, "comfort")

        if return_trunk_features:
            out["trunk_features"] = trunk  # [B, K, H]

        return out
