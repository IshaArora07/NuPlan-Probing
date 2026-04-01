from copy import deepcopy
import math

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .layers.mlp_layer import MLPLayer

# ---------------- EMoE imports ----------------
from .EMOE_Planner.scene_router import SceneRouter
from .EMOE_Planner.scene_mode_query import SceneModeQueryGeneratorHard
from .EMOE_Planner.planning_decoder import EMoEPlanningDecoder
from .EMOE_Planner.interaction_pred_decoder import InteractionPredDecoder

trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,                 # Ka
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),

        # -------- EMoE --------
        num_scene_types: int = 6,     
        interaction_pred_output_dim_per_step: int = 2,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.num_scene_types = num_scene_types
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        # ---------------- Encoders (PLUTO) ----------------
        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # ---------------- EMoE ----------------
        self.scene_router = SceneRouter(
            d_model=dim,
            num_scene_types=num_scene_types,
            hidden_dim=dim,
            dropout=dropout,
            use_token_pooling=True,
        )

        scene_anchors_path = "......"
        anchors = np.load(scene_anchors_path)
        anchors_xy_init = torch.from_numpy(anchors).float()
        self.mode_query_generator = SceneModeQueryGeneratorHard(
            anchors_xy=anchors_xy_init,
            d_model=dim,
            num_fourier_bands=8,
            use_scene_type_embed=True,
        )

        self.planning_decoder = EMoEPlanningDecoder(
            d_model=dim,
            nhead=num_heads,
            dim_ff=4 * dim,
            future_steps=future_steps,
            num_layers=decoder_depth,
            num_experts=num_scene_types,
            dropout_p=dropout,
        )

        self.interaction_pred_decoder = InteractionPredDecoder(
            d_model=dim,
            nhead=num_heads,
            dim_ff=4 * dim,
            num_layers=2,
            T_pred=future_steps,
            dropout_p=dropout,
            use_agent_self_attn=True,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ======================================================
    # ====================== FORWARD =======================
    # ======================================================
    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)
        pos = torch.cat([pos, static_pos], dim=1)
        x = x + self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        # ---------------- HARD ROUTER ----------------
        scene_logits, scene_idx = self.scene_router(scene_tokens=x)

        # ---------------- HARD QUERY GEN ----------------
        mode_queries = self.mode_query_generator(scene_labels=scene_idx)

        # ---------------- HARD EMoE DECODER ----------------
        decoded_queries, traj_modes, scores_modes = self.planning_decoder(
            mode_queries=mode_queries,
            scene_tokens=x,
            scene_idx=scene_idx,
            key_padding_mask=key_padding_mask,
        )

        trajectory = traj_modes.unsqueeze(1)    # [B, 1, Ka, T, 6]
        probability = scores_modes.unsqueeze(1)

        agent_tokens = x[:, 1:A]
        agent_padding_mask = agent_key_padding[:, 1:A]

        prediction = self.interaction_pred_decoder(
            agent_tokens=agent_tokens,
            scene_tokens=x,
            ego_mode_queries=decoded_queries,
            agent_padding_mask=agent_padding_mask,
            scene_padding_mask=key_padding_mask,
        )

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "router_logits": scene_logits,
            "router_idx": scene_idx,
            "mode_queries": decoded_queries,
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        return out
