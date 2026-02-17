import logging
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    """
    HARD-EMoE Lightning Trainer.

    Loss:
        L = w_reg * L_reg
          + w_cls * L_cls
          + w_col * L_col
          + w_pred * L_pred
          + w_router * L_router
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        use_collision_loss: bool = True,
        regulate_yaw: bool = False,
        objective_aggregate_mode: str = "mean",

        # EMoE loss weights
        k_R: float = 0.2,
        interaction_distance_threshold: float = 8.0,
        w_reg: float = 1.0,
        w_cls: float = 1.0,
        w_col: float = 1.0,
        w_pred: float = 1.0,
        w_router: float = 1.0,
    ) -> None:
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

        self.use_collision_loss = use_collision_loss
        self.regulate_yaw = regulate_yaw

        self.k_R = k_R
        self.interaction_distance_threshold = interaction_distance_threshold

        self.w_reg = w_reg
        self.w_cls = w_cls
        self.w_col = w_col
        self.w_pred = w_pred
        self.w_router = w_router

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def forward(self, features: FeaturesType):
        return self.model(features)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        prefix: str,
    ):
        features, targets, scenarios = batch
        data = features["feature"].data

        res = self.forward(data)

        losses = self._compute_objectives(res, data)
        metrics = self._compute_metrics(res, data, prefix)

        self._log_step(losses["loss"], losses, metrics, prefix)
        return losses["loss"] if self.training else 0.0

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        """
        HARD-EMoE objective computation.
        """

        trajectory = res["trajectory"]          # [B, 1, Ka, T, 6]
        probability = res["probability"]        # [B, 1, Ka]
        prediction = res["prediction"]          # [B, A-1, T, 2]

        bs = trajectory.shape[0]
        T = prediction.shape[2]

        # -------------------------
        # Ground truth
        # -------------------------
        targets_pos = data["agent"]["target"][:bs]
        valid_mask = data["agent"]["valid_mask"][:bs, :, -T:]
        targets_vel = data["agent"]["velocity"][:bs, :, -T:]

        target_xy = targets_pos[..., -T:, :2]
        target_yaw = targets_pos[..., -T:, 2]
        target_cos_sin = torch.stack(
            [target_yaw.cos(), target_yaw.sin()], dim=-1
        )

        target_6d = torch.cat(
            [target_xy, target_cos_sin, targets_vel], dim=-1
        )

        ego_valid_mask = valid_mask[:, 0]
        ego_target_6d = target_6d[:, 0]

        pred_valid_mask = valid_mask[:, 1:]
        pred_target_xy = target_xy[:, 1:]

        # -------------------------
        # Planning losses
        # -------------------------
        ego_reg_loss, ego_cls_loss, collision_loss = self.get_planning_loss(
            data,
            trajectory,
            probability,
            ego_valid_mask,
            ego_target_6d,
            bs,
        )

        # -------------------------
        # Prediction loss
        # -------------------------
        prediction_loss = self.get_prediction_loss(
            prediction, pred_valid_mask, pred_target_xy
        )

        # -------------------------
        # Router loss (HARD)
        # -------------------------
        router_logits = res["router_logits"]
        scene_labels = data["emoe"]["scene_label"][:bs].long()

        router_loss = F.cross_entropy(router_logits, scene_labels)

        # -------------------------
        # Total loss
        # -------------------------
        loss = (
            self.w_reg * ego_reg_loss
            + self.w_cls * ego_cls_loss
            + self.w_col * collision_loss
            + self.w_pred * prediction_loss
            + self.w_router * router_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.detach(),
            "cls_loss": ego_cls_loss.detach(),
            "collision_loss": collision_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "router_loss": router_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Prediction loss
    # ------------------------------------------------------------------

    def get_prediction_loss(self, prediction, valid_mask, target_xy):
        diff = F.smooth_l1_loss(
            prediction, target_xy, reduction="none"
        ).sum(-1)

        diff = diff * valid_mask
        return diff.sum() / (valid_mask.sum() + 1e-6)

    # ------------------------------------------------------------------
    # Planning losses (unchanged PLUTO logic)
    # ------------------------------------------------------------------

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(0, 7)

        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)

        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )

        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0]
            / (self.radius / self.num_modes)
        ).long().clamp_(0, self.num_modes - 1)

        best_trajectory = trajectory[
            torch.arange(bs), target_r_index, target_m_index
        ]

        # -------------------------
        # Collision loss
        # -------------------------
        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        # -------------------------
        # Regression loss
        # -------------------------
        reg_per_step = F.smooth_l1_loss(
            best_trajectory, target, reduction="none"
        ).sum(-1)

        time_idx = torch.arange(reg_per_step.shape[1], device=reg_per_step.device)
        weights = torch.exp(-self.k_R * time_idx).unsqueeze(0)

        weighted_mask = weights * valid_mask
        reg_loss = (reg_per_step * weighted_mask).sum() / (
            weighted_mask.sum() + 1e-6
        )

        # -------------------------
        # Classification loss
        # -------------------------
        probability = probability.reshape(bs, -1)
        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_m_index] = 1.0

        cls_loss = F.cross_entropy(probability, target_label.detach())

        return reg_loss, cls_loss, collision_loss

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, res, data, prefix):
        trajectory = res["trajectory"]
        probability = res["probability"]

        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)

        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        return self.metrics[prefix](outputs, target)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, loss, objectives, metrics, prefix):
        self.log(
            f"loss/{prefix}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=(prefix == "train"),
        )

        for k, v in objectives.items():
            self.log(
                f"objectives/{prefix}_{k}",
                v,
                on_epoch=True,
                sync_dist=True,
            )

        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=(prefix == "val"),
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = WarmupCosLR(
            optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
