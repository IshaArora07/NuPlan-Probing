import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneModeQueryGeneratorHard(nn.Module):
    """
    Hard-routing version of the Scene-specific mode query generator.

    Implements EMoEPlanner's HARD routing:
        1. Pick a single scene-type s* = argmax p_scene[b]
        2. Use ONLY that scene type's anchor set G_{s*} for the queries
        3. Q_mode = Q_learnable + FOPE(G_{s*})

    Inputs:
        anchors_xy:  [S, Ka, 2]
        scene_probs: [B, S]  (logits or probabilities)
        scene_labels: [B]    (used only if scene_probs not provided)

    Returns:
        mode_queries: [B, Ka, d_model]
    """

    def __init__(
        self,
        anchors_xy: torch.Tensor,
        d_model: int,
        num_fourier_bands: int = 8,
        use_scene_type_embed: bool = True,
    ):
        super().__init__()

        assert anchors_xy.ndim == 3 and anchors_xy.shape[-1] == 2, \
            "anchors_xy should be [num_scene_types, Ka, 2]"

        num_scene_types, Ka, _ = anchors_xy.shape

        self.num_scene_types = num_scene_types  # e.g. 7
        self.num_anchors = Ka
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands

        # Non-trainable anchor table
        self.register_buffer("anchors_xy", anchors_xy.float(), persistent=True)

        # Frequencies for FOPE
        freqs = 2.0 ** torch.arange(num_fourier_bands).float()
        self.register_buffer("fourier_freqs", freqs, persistent=True)

        fope_dim = 4 * num_fourier_bands
        self.fope_proj = nn.Linear(fope_dim, d_model)

        # Learnable base queries (same as soft variant)
        self.Q_learnable = nn.Parameter(
            torch.randn(self.num_anchors, d_model) / math.sqrt(d_model)
        )

        self.use_scene_type_embed = use_scene_type_embed
        if use_scene_type_embed:
            self.scene_type_embed = nn.Embedding(num_scene_types, d_model)

        self.query_ln = nn.LayerNorm(d_model)

    def _fope_encode(self, coords_xy: torch.Tensor) -> torch.Tensor:
        """
        FOPE encoding for 2D (x, y) coordinates.

        coords_xy: [B, Ka, 2]
        returns:    [B, Ka, 4F]
        """
        x = coords_xy[..., 0]  # [B, Ka]
        y = coords_xy[..., 1]  # [B, Ka]

        freqs = self.fourier_freqs  # [F]

        x_exp = x.unsqueeze(-1) * freqs
        y_exp = y.unsqueeze(-1) * freqs

        fope = torch.cat(
            [torch.sin(x_exp), torch.cos(x_exp),
             torch.sin(y_exp), torch.cos(y_exp)],
            dim=-1
        )
        return fope  # [B, Ka, 4F]

    def forward(
        self,
        scene_probs: Optional[torch.Tensor] = None,
        scene_labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # ---------------------------
        # 1. Determine the hard scene-type index
        # ---------------------------
        if scene_probs is not None:
            # Convert to probability distribution
            probs = F.softmax(scene_probs, dim=-1)
            # Hard selection
            hard_ids = torch.argmax(probs, dim=-1)  # [B]
        else:
            # Use provided hard labels
            assert scene_labels is not None, "Provide scene_probs or scene_labels"
            hard_ids = scene_labels  # [B]

        B = hard_ids.shape[0]

        # ---------------------------
        # 2. Select anchors for each batch element
        # ---------------------------
        # anchors_xy: [S, Ka, 2]
        anchors = self.anchors_xy[hard_ids]  # [B, Ka, 2]

        # ---------------------------
        # 3. FOPE encode the chosen anchors
        # ---------------------------
        fope_feats = self._fope_encode(anchors)  # [B, Ka, 4F]
        fope_proj = self.fope_proj(fope_feats)   # [B, Ka, d_model]

        # ---------------------------
        # 4. Add learnable base queries
        # ---------------------------
        Q_base = self.Q_learnable.unsqueeze(0).expand(B, -1, -1)
        mode_queries = Q_base + fope_proj

        # ---------------------------
        # 5. Optional scene type embedding
        # ---------------------------
        if self.use_scene_type_embed:
            scene_bias = self.scene_type_embed(hard_ids)  # [B, d_model]
            scene_bias = scene_bias.unsqueeze(1)          # [B, 1, d_model]
            mode_queries = mode_queries + scene_bias

        # ---------------------------
        # 6. LayerNorm for stability
        # ---------------------------
        mode_queries = self.query_ln(mode_queries)
        return mode_queries
