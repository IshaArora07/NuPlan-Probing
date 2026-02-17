import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneRouter(nn.Module):
    """
    HARD Scene Router for EMoE.

    Outputs:
        - scene_logits: [B, S]   (for CE loss)
        - scene_idx:    [B]      (hard routing decision)

    No soft routing is used downstream.
    """

    def __init__(
        self,
        d_model: int,
        num_scene_types: int = 7,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_token_pooling: bool = True,
        pool_fn: str = "ego",   # {"ego", "mean"}
    ):
        super().__init__()

        self.d_model = d_model
        self.num_scene_types = num_scene_types
        self.hidden_dim = hidden_dim or d_model
        self.use_token_pooling = use_token_pooling
        self.pool_fn = pool_fn

        # Normalize pooled scene embedding
        self.input_ln = nn.LayerNorm(d_model)

        # Router MLP
        self.router_mlp = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Logits over scene types
        self.scene_head = nn.Linear(self.hidden_dim, num_scene_types)

    # ----------------------------------------------------------
    # Scene token pooling
    # ----------------------------------------------------------
    def _pool_scene_tokens(
        self,
        scene_tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, L_ctx, d_model]
            padding_mask: [B, L_ctx] where True = padding

        Returns:
            scene_global: [B, d_model]
        """
        if self.pool_fn == "ego":
            # PLUTO convention: token 0 is ego / global token
            return scene_tokens[:, 0]

        elif self.pool_fn == "mean":
            if padding_mask is None:
                return scene_tokens.mean(dim=1)

            valid = ~padding_mask
            denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
            return (scene_tokens * valid.unsqueeze(-1)).sum(dim=1) / denom

        else:
            raise ValueError(f"Unknown pool_fn={self.pool_fn}")

    # ----------------------------------------------------------
    # Forward (HARD routing)
    # ----------------------------------------------------------
    def forward(
        self,
        scene_tokens: torch.Tensor | None = None,
        scene_padding_mask: torch.Tensor | None = None,
        scene_global: torch.Tensor | None = None,
    ):
        """
        Returns:
            scene_logits: [B, S]   (for CE loss)
            scene_idx:    [B]      (hard routing decision)
        """
        # 1. Obtain global scene embedding
        if scene_global is None:
            assert scene_tokens is not None, \
                "Provide scene_tokens or scene_global"
            assert self.use_token_pooling, \
                "Router configured without pooling; pass scene_global"

            scene_global = self._pool_scene_tokens(
                scene_tokens,
                scene_padding_mask,
            )

        # 2. Normalize
        x = self.input_ln(scene_global)  # [B, d_model]

        # 3. Router MLP
        h = self.router_mlp(x)           # [B, hidden_dim]

        # 4. Logits
        scene_logits = self.scene_head(h)  # [B, S]

        # 5. HARD routing decision
        scene_idx = torch.argmax(scene_logits, dim=-1)  # [B]

        return scene_logits, scene_idx
