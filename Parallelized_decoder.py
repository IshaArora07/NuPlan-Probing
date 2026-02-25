import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.mlp_layer import MLPLayer


# ------------------------------------------------------------
# Expert FFN
# ------------------------------------------------------------
class EMoEFFNExpert(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ------------------------------------------------------------
# HARD EMoE Decoder Layer
# ------------------------------------------------------------
class EMoEDecoderLayer(nn.Module):
    """
    HARD EMoE decoder layer:
      - Exactly ONE expert per batch element
      - No mixture, no soft routing
      - Vectorized expert dispatch: loop over experts not batch elements
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_experts: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # 1) Self-attention over mode queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout_p,
            batch_first=True,
        )

        # 2) Cross-attention to scene tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout_p,
            batch_first=True,
        )

        # Norms
        self.ln_self = nn.LayerNorm(d_model)
        self.ln_cross = nn.LayerNorm(d_model)
        self.ln_moe = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout_self = nn.Dropout(dropout_p)
        self.dropout_cross = nn.Dropout(dropout_p)
        self.dropout_moe = nn.Dropout(dropout_p)

        # Experts
        self.experts = nn.ModuleList(
            [EMoEFFNExpert(d_model, dim_ff, dropout_p) for _ in range(num_experts)]
        )

    def forward(
        self,
        mode_queries: torch.Tensor,      # [B, Ka, D]
        scene_tokens: torch.Tensor,      # [B, L, D]
        scene_idx: torch.LongTensor,     # [B]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Ka, D = mode_queries.shape
        x = mode_queries

        # ---- 1) Self-attention ----
        q = self.ln_self(x)
        sa_out, _ = self.self_attn(
            query=q,
            key=q,
            value=q,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout_self(sa_out)

        # ---- 2) Cross-attention ----
        q = self.ln_cross(x)
        ca_out, _ = self.cross_attn(
            query=q,
            key=scene_tokens,
            value=scene_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout_cross(ca_out)

        # ---- 3) Vectorized HARD expert FFN ----
        # Loop over experts (num_experts iterations) rather than batch
        # elements (B iterations). Each expert processes its assigned
        # batch elements in one batched forward pass.
        q = self.ln_moe(x)
        out = torch.zeros_like(q)

        for expert_id in range(self.num_experts):
            mask = (scene_idx == expert_id)  # [B] bool
            if not mask.any():
                continue
            # q[mask]: [n, Ka, D] where n = number of batch elements
            # assigned to this expert. Processed in one call.
            out[mask] = self.experts[expert_id](q[mask])

        x = x + self.dropout_moe(out)
        return x


# ------------------------------------------------------------
# HARD EMoE Planning Decoder
# ------------------------------------------------------------
class EMoEPlanningDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        future_steps: int,
        num_layers: int,
        num_experts: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                EMoEDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_ff=dim_ff,
                    num_experts=num_experts,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

        # PLUTO-style heads
        self.loc_head = MLPLayer(d_model, 2 * d_model, future_steps * 2)
        self.yaw_head = MLPLayer(d_model, 2 * d_model, future_steps * 2)
        self.vel_head = MLPLayer(d_model, 2 * d_model, future_steps * 2)
        self.pi_head = MLPLayer(d_model, d_model, 1)

    def forward(
        self,
        mode_queries: torch.Tensor,      # [B, Ka, D]
        scene_tokens: torch.Tensor,      # [B, L, D]
        scene_idx: torch.LongTensor,     # [B]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        x = mode_queries

        for layer in self.layers:
            x = layer(
                mode_queries=x,
                scene_tokens=scene_tokens,
                scene_idx=scene_idx,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        decoded_queries = x  # [B, Ka, D]

        B, Ka, D = decoded_queries.shape

        loc = self.loc_head(decoded_queries).view(B, Ka, self.future_steps, 2)
        yaw = self.yaw_head(decoded_queries).view(B, Ka, self.future_steps, 2)
        vel = self.vel_head(decoded_queries).view(B, Ka, self.future_steps, 2)

        scores = self.pi_head(decoded_queries).squeeze(-1)  # [B, Ka]

        traj = torch.cat([loc, yaw, vel], dim=-1)  # [B, Ka, T, 6]

        return decoded_queries, traj, scores
