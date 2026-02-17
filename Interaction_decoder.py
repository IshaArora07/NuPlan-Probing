import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InteractionPredDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout_p: float = 0.1,
        use_agent_self_attn: bool = True,
    ):
        super().__init__()
        self.use_agent_self_attn = use_agent_self_attn

        if use_agent_self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout_p, batch_first=True
            )
            self.ln_self = nn.LayerNorm(d_model)
            self.dropout_self = nn.Dropout(dropout_p)

        self.cross_attn_scene = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_p, batch_first=True
        )
        self.ln_scene = nn.LayerNorm(d_model)
        self.dropout_scene = nn.Dropout(dropout_p)

        self.cross_attn_ego = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_p, batch_first=True
        )
        self.ln_ego = nn.LayerNorm(d_model)
        self.dropout_ego = nn.Dropout(dropout_p)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropout_ffn = nn.Dropout(dropout_p)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        ego_mode_queries: torch.Tensor,
        agent_padding_mask: Optional[torch.Tensor] = None,
        scene_padding_mask: Optional[torch.Tensor] = None,
    ):
        x = agent_tokens

        if self.use_agent_self_attn:
            q = self.ln_self(x)
            sa, _ = self.self_attn(
                q, q, q, key_padding_mask=agent_padding_mask, need_weights=False
            )
            x = x + self.dropout_self(sa)

        q = self.ln_scene(x)
        ca, _ = self.cross_attn_scene(
            q, scene_tokens, scene_tokens,
            key_padding_mask=scene_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout_scene(ca)

        q = self.ln_ego(x)
        ca, _ = self.cross_attn_ego(
            q, ego_mode_queries, ego_mode_queries, need_weights=False
        )
        x = x + self.dropout_ego(ca)

        q = self.ln_ffn(x)
        f = self.fc1(q)
        f = F.relu(f, inplace=True)
        f = self.dropout_ffn(f)
        f = self.fc2(f)
        f = self.dropout_ffn(f)
        x = x + f

        return x


class InteractionPredDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_layers: int,
        T_pred: int,
        dropout_p: float = 0.1,
        use_agent_self_attn: bool = True,
    ):
        super().__init__()
        self.T_pred = T_pred
        self.output_dim_per_step = 2

        self.layers = nn.ModuleList(
            [
                InteractionPredDecoderLayer(
                    d_model, nhead, dim_ff, dropout_p, use_agent_self_attn
                )
                for _ in range(num_layers)
            ]
        )

        self.traj_head = nn.Linear(d_model, T_pred * self.output_dim_per_step)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        ego_mode_queries: torch.Tensor,
        agent_padding_mask: Optional[torch.Tensor] = None,
        scene_padding_mask: Optional[torch.Tensor] = None,
    ):
        x = agent_tokens

        for layer in self.layers:
            x = layer(
                agent_tokens=x,
                scene_tokens=scene_tokens,
                ego_mode_queries=ego_mode_queries,
                agent_padding_mask=agent_padding_mask,
                scene_padding_mask=scene_padding_mask,
            )

        traj = self.traj_head(x)
        B, N, _ = traj.shape
        return traj.view(B, N, self.T_pred, self.output_dim_per_step)
