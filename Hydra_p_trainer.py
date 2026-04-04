# pluto_trainer.py

import json
import logging
import os
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss
from .EMOE_Planner.hydra_losses import HydraLosses, HydraLossConfig
from .EMOE_Planner.rule_based_teachers import (
    RuleBasedTeachers,
    RuleTeacherConfig,
)

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        use_collision_loss: bool = True,
        regulate_yaw: bool = False,
        k_R: float = 0.2,
        interaction_distance_threshold: float = 8.0,
        w_reg: float = 1.0,
        w_cls: float = 2.0,
        w_col: float = 1.0,
        w_pred: float = 1.0,
        w_router: float = 1.0,
        gmm_sigma: float = 1.0,
        w_diversity: float = 0.05,
        diversity_min_sep: float = 0.5,
        debug_router_steps: int = 5,
        use_contrast_loss: bool = False,
        use_hydra_aux: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

        self.history_steps = model.history_steps
        self.radius = model.radius
        self.num_modes = model.num_modes
        self.num_scene_types = model.num_scene_types

        self.use_collision_loss = use_collision_loss
        self.regulate_yaw = regulate_yaw
        self.k_R = k_R
        self.interaction_distance_threshold = interaction_distance_threshold

        self.w_reg = w_reg
        self.w_cls = w_cls
        self.w_col = w_col
        self.w_pred = w_pred
        self.w_router = w_router

        self.gmm_sigma = gmm_sigma
        self.w_diversity = w_diversity
        self.diversity_min_sep = diversity_min_sep

        self.debug_router_steps = debug_router_steps
        self.use_contrast_loss = use_contrast_loss
        self.use_hydra_aux = use_hydra_aux

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

        labels_path = os.environ.get("LABELS_PATH", None)
        if labels_path is not None:
            router_class_weights = self._compute_router_weights(
                labels_path,
                self.num_scene_types,
            )
        else:
            router_class_weights = (
                torch.ones(self.num_scene_types, dtype=torch.float32)
                / float(self.num_scene_types)
            )

        self.register_buffer("router_class_weights", router_class_weights)

        # ======================================================
        # Hydra auxiliary supervision
        # ======================================================
        if self.use_hydra_aux:
            teacher_cfg = RuleTeacherConfig()
            self.rule_teachers = RuleBasedTeachers(teacher_cfg)

            hydra_cfg = HydraLossConfig(
                w_feasibility=0.03,
                w_cost=0.05,
                w_progress=0.01,
                w_comfort=0.01,
            )
            self.hydra_losses = HydraLosses(hydra_cfg)
        else:
            self.rule_teachers = None
            self.hydra_losses = None

    @staticmethod
    def _compute_router_weights(
        scene_labels_path: str,
        num_classes: int,
    ) -> torch.Tensor:
        counts = torch.zeros(num_classes, dtype=torch.float32)

        with open(scene_labels_path, "r") as f:
            for line in f:
                cls = json.loads(line).get("emoe_class_id", -1)
                if 0 <= cls < num_classes:
                    counts[cls] += 1

        weights = 1.0 / counts.clamp(min=1)
        weights = weights / weights.sum()
        return weights

    def forward(self, features: FeaturesType):
        return self.model(features)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        prefix: str,
    ):
        features, targets, scenarios = batch
        data = features["feature"].data

        res = self.forward(data)
        losses = self._compute_objectives(res, data, prefix)

        self.log_dict(
            {f"{prefix}/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=(prefix == "train"),
        )

        return losses["loss"]

    def _compute_objectives(
        self,
        res,
        data,
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        bs, _, t_steps, _ = res["prediction"].shape

        trajectory = res["trajectory"]
        probability = res["probability"]
        prediction = res["prediction"]

        targets_pos = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -t_steps:]
        targets_vel = data["agent"]["velocity"][:, :, -t_steps:]

        target_xy = targets_pos[..., -t_steps:, :2]
        target_yaw = targets_pos[..., -t_steps:, 2]
        target_cos_sin = torch.stack(
            [target_yaw.cos(), target_yaw.sin()],
            dim=-1,
        )
        target_6d = torch.cat(
            [target_xy, target_cos_sin, targets_vel],
            dim=-1,
        )

        ego_valid_mask = valid_mask[:, 0]
        ego_target_6d = target_6d[:, 0]

        pred_valid_mask = valid_mask[:, 1:]
        pred_target_xy = target_xy[:, 1:]

        (
            ego_reg_loss,
            ego_cls_loss,
            collision_loss,
            diversity_loss,
        ) = self.get_planning_loss(
            data,
            trajectory,
            probability,
            ego_valid_mask,
            ego_target_6d,
            bs,
        )

        prediction_loss = self.get_prediction_loss(
            prediction,
            pred_valid_mask,
            pred_target_xy,
        )

        router_logits = res["router_logits"]
        scene_labels = data["emoe"]["emoe_class_id"][:bs].long()

        router_ce = F.cross_entropy(
            router_logits,
            scene_labels,
            weight=self.router_class_weights,
        )

        loss = (
            self.w_reg * ego_reg_loss
            + self.w_cls * ego_cls_loss
            + self.w_col * collision_loss
            + self.w_pred * prediction_loss
            + self.w_router * router_ce
            + self.w_diversity * diversity_loss
        )

        # ======================================================
        # Hydra auxiliary losses
        # ======================================================
        hydra_total = torch.zeros((), device=loss.device)
        hydra_losses_dict: Dict[str, torch.Tensor] = {}
        teacher: Dict[str, torch.Tensor] = {}

        if (
            self.use_hydra_aux
            and self.rule_teachers is not None
            and self.hydra_losses is not None
        ):
            hydra_out = res.get("hydra_heads", None)

            if hydra_out is not None:
                traj_all_modes = trajectory[:, 0]

                teacher = self.rule_teachers(
                    traj_all_modes,
                    context=None,
                )

                hydra_losses_dict = self.hydra_losses(
                    hydra_out,
                    teacher,
                )

                hydra_total = hydra_losses_dict["loss_hydra_total"]
                loss = loss + hydra_total

        feasible_rate = torch.zeros((), device=loss.device)
        if teacher and "feasibility" in teacher:
            feasible_rate = (
                teacher["feasibility"] >= 0.5
            ).float().mean()

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.detach(),
            "cls_loss": ego_cls_loss.detach(),
            "collision_loss": collision_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "router_loss": router_ce.detach(),
            "diversity_loss": diversity_loss.detach(),
            "hydra_total": hydra_total.detach(),
            "hydra_loss_feasibility": hydra_losses_dict.get(
                "loss_feasibility",
                torch.zeros((), device=loss.device),
            ).detach(),
            "hydra_loss_cost": hydra_losses_dict.get(
                "loss_cost",
                torch.zeros((), device=loss.device),
            ).detach(),
            "hydra_loss_progress": hydra_losses_dict.get(
                "loss_progress",
                torch.zeros((), device=loss.device),
            ).detach(),
            "hydra_loss_comfort": hydra_losses_dict.get(
                "loss_comfort",
                torch.zeros((), device=loss.device),
            ).detach(),
            "teacher_feasible_rate": feasible_rate.detach(),
        }

    def configure_optimizers(self):
        scale_params = [
            p
            for n, p in self.model.named_parameters()
            if "score_attn_scale" in n
        ]
        other_params = [
            p
            for n, p in self.model.named_parameters()
            if "score_attn_scale" not in n
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": other_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": scale_params,
                    "lr": self.lr * 0.05,
                    "weight_decay": 0.0,
                },
            ]
        )

        scheduler = WarmupCosLR(
            optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
