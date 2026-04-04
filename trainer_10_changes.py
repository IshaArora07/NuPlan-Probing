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

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    """
    HARD-EMoE Lightning Trainer.

    Must-have changes (correctness fixes):
      1. score_attn_scale gets lr * 0.05 in configure_optimizers
      2. WTA uses xy-only for mode selection
      3. Router CE uses inverse-frequency class weights from LABELS_PATH
      4. Gradient clipping: add gradient_clip_val=1.0 to Lightning trainer yaml
      5. Endpoint loss added directly into reg_loss
      6. Temporal reweighting blends decay with endpoint emphasis
      7. w_cls default raised from 1.0 to 2.0

    Medium-impact changes:
      8. GMM loss replaces hard cross-entropy cls_loss for mode scoring
      9. Mode diversity loss penalises endpoint clustering across Ka modes
     10. Ka=32 — set num_modes=32 in model config (no code change here)
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

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

        self._router_stats_train = None
        self._router_stats_val = None

        labels_path = os.environ.get("LABELS_PATH", None)
        if labels_path is not None:
            router_class_weights = self._compute_router_weights(
                labels_path, self.num_scene_types
            )
            logger.info(f"[EMoE] Loaded router class weights from {labels_path}")
        else:
            logger.warning(
                "[EMoE] LABELS_PATH not set — using uniform router class weights."
            )
            router_class_weights = torch.ones(
                self.num_scene_types, dtype=torch.float32
            ) / float(self.num_scene_types)

        self.register_buffer("router_class_weights", router_class_weights)

    @staticmethod
    def _compute_router_weights(scene_labels_path: str, num_classes: int) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from scene_labels.jsonl.
        """
        counts = torch.zeros(num_classes, dtype=torch.float32)
        with open(scene_labels_path, "r") as f:
            for line in f:
                cls = json.loads(line).get("emoe_class_id", -1)
                if 0 <= cls < num_classes:
                    counts[cls] += 1

        logger.info(f"[EMoE] Router class counts: {counts.long().tolist()}")
        weights = 1.0 / counts.clamp(min=1)
        weights = weights / weights.sum()
        logger.info(f"[EMoE] Router class weights: {weights.tolist()}")
        return weights

    def on_fit_start(self):
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

    def _make_router_stats(self):
        s = self.num_scene_types
        return {
            "count": 0,
            "correct": 0,
            "entropy_sum": 0.0,
            "label_counts": torch.zeros(s, device=self.device),
            "expert_counts": torch.zeros(s, device=self.device),
            "confusion": torch.zeros(s, s, device=self.device),
        }

    def _get_router_stats(self, prefix: str):
        if prefix == "train":
            return self._router_stats_train
        if prefix == "val":
            return self._router_stats_val
        return None

    def on_train_epoch_start(self):
        self._router_stats_train = self._make_router_stats()

    def on_validation_epoch_start(self):
        self._router_stats_val = self._make_router_stats()

    def _log_router_stats(self, prefix: str):
        stats = self._get_router_stats(prefix)
        if stats is None or stats["count"] == 0:
            return

        count = float(stats["count"])
        entropy_mean = float(stats["entropy_sum"]) / count
        accuracy = float(stats["correct"]) / count

        self.log(f"router/{prefix}_entropy", entropy_mean, prog_bar=True, on_epoch=True)
        self.log(f"router/{prefix}_accuracy", accuracy, prog_bar=True, on_epoch=True)

        usage = stats["expert_counts"] / (stats["expert_counts"].sum() + 1e-6)
        for i in range(self.num_scene_types):
            self.log(f"router/{prefix}_usage_expert_{i}", usage[i], on_epoch=True)
            self.log(
                f"router/{prefix}_label_count_{i}",
                stats["label_counts"][i],
                on_epoch=True,
            )

        row_denom = stats["label_counts"].clamp(min=1).unsqueeze(1)
        conf_norm = stats["confusion"] / row_denom
        for i in range(self.num_scene_types):
            for j in range(self.num_scene_types):
                self.log(
                    f"router/{prefix}_confusion_{i}_to_{j}",
                    conf_norm[i, j],
                    on_epoch=True,
                )

    def on_train_epoch_end(self):
        self._log_router_stats("train")

    def on_validation_epoch_end(self):
        self._log_router_stats("val")

    def forward(self, features: FeaturesType):
        return self.model(features)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        prefix: str,
    ):
        features, targets, scenarios = batch
        data = features["feature"].data

        res = self.forward(data)
        losses = self._compute_objectives(res, data, prefix=prefix)
        metrics = self._compute_metrics(res, data, prefix)

        self._log_step(losses["loss"], losses, metrics, prefix)
        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data, prefix: str) -> Dict[str, torch.Tensor]:
        bs, _, t_steps, _ = res["prediction"].shape

        train_num = bs
        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs

        trajectory = res["trajectory"][:train_num]
        probability = res["probability"][:train_num]
        prediction = res["prediction"][:train_num]

        targets_pos = data["agent"]["target"][:train_num]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -t_steps:]
        targets_vel = data["agent"]["velocity"][:train_num, :, -t_steps:]

        target_xy = targets_pos[..., -t_steps:, :2]
        target_yaw = targets_pos[..., -t_steps:, 2]
        target_cos_sin = torch.stack([target_yaw.cos(), target_yaw.sin()], dim=-1)
        target_6d = torch.cat([target_xy, target_cos_sin, targets_vel], dim=-1)

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
            data, trajectory, probability, ego_valid_mask, ego_target_6d, train_num
        )

        prediction_loss = self.get_prediction_loss(
            prediction, pred_valid_mask, pred_target_xy
        )

        router_logits = res["router_logits"][:train_num]
        n = router_logits.shape[0]
        scene_labels = data["emoe"]["emoe_class_id"][:n].long()

        router_ce = F.cross_entropy(
            router_logits,
            scene_labels,
            weight=self.router_class_weights,
        )

        with torch.no_grad():
            preds = torch.argmax(router_logits, dim=-1)
            probs = F.softmax(router_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

            stats = self._get_router_stats(prefix)
            if stats is not None:
                s = self.num_scene_types
                stats["count"] += int(n)
                stats["correct"] += int((preds == scene_labels).sum().item())
                stats["entropy_sum"] += float(entropy.sum().item())
                stats["label_counts"] += torch.bincount(
                    scene_labels, minlength=s
                ).to(stats["label_counts"].dtype)
                stats["expert_counts"] += torch.bincount(
                    preds, minlength=s
                ).to(stats["expert_counts"].dtype)
                stats["confusion"].index_put_(
                    (scene_labels, preds),
                    torch.ones_like(scene_labels, dtype=stats["confusion"].dtype),
                    accumulate=True,
                )

            ce_true = (
                -F.log_softmax(router_logits, dim=-1)
                .gather(1, scene_labels.view(-1, 1))
                .squeeze(1)
                .mean()
            )
            acc_true = (preds == scene_labels).float().mean()

            self.log(
                f"router/{prefix}_ce_true",
                ce_true,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"router/{prefix}_acc_true",
                acc_true,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"router/{prefix}_num_unique_preds",
                float(torch.unique(preds).numel()),
                on_step=True,
                on_epoch=True,
            )

            if self.global_step < self.debug_router_steps:
                logger.warning(
                    f"[Router debug] prefix={prefix} step={self.global_step} "
                    f"labels={scene_labels.detach().cpu().tolist()} "
                    f"preds={preds.detach().cpu().tolist()} "
                    f"CE={router_ce.item():.6f} acc={acc_true.item():.3f} "
                    f"unique_preds={int(torch.unique(preds).numel())}"
                )

        loss = (
            self.w_reg * ego_reg_loss
            + self.w_cls * ego_cls_loss
            + self.w_col * collision_loss
            + self.w_pred * prediction_loss
            + self.w_router * router_ce
            + self.w_diversity * diversity_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.detach(),
            "cls_loss": ego_cls_loss.detach(),
            "collision_loss": collision_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "router_loss": router_ce.detach(),
            "diversity_loss": diversity_loss.detach(),
        }

    def get_prediction_loss(self, prediction, valid_mask, target_xy):
        diff = F.smooth_l1_loss(prediction, target_xy, reduction="none").sum(-1)
        diff = diff * valid_mask.float()
        return diff.sum() / (valid_mask.float().sum() + 1e-6)

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        assert trajectory.dim() == 5, trajectory.shape
        assert probability.dim() == 3, probability.shape
        assert trajectory.shape[0] == bs and probability.shape[0] == bs
        assert trajectory.shape[1] == 1 and probability.shape[1] == 1, (
            f"EMoE expects R=1 but got traj={trajectory.shape}, prob={probability.shape}"
        )

        ka = trajectory.shape[2]
        t_steps = trajectory.shape[3]

        traj_modes = trajectory[:, 0]
        prob_logits = probability[:, 0]

        mask = valid_mask.float()[:, None, :]

        per_step = F.smooth_l1_loss(
            traj_modes[..., :2],
            target[:, None, :, :2],
            reduction="none",
        ).sum(-1)
        per_mode = (per_step * mask).sum(-1) / (mask.sum(-1) + 1e-6)
        target_m_index = torch.argmin(per_mode, dim=-1)

        best_trajectory = traj_modes[
            torch.arange(bs, device=traj_modes.device), target_m_index
        ]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = traj_modes.new_zeros(())

        reg_per_step = F.smooth_l1_loss(
            best_trajectory, target, reduction="none"
        ).sum(-1)

        time_idx = torch.arange(
            t_steps,
            device=reg_per_step.device,
            dtype=reg_per_step.dtype,
        )
        decay_w = torch.exp(-self.k_R * time_idx)
        endpoint_w = torch.linspace(
            0.5, 1.5, t_steps, device=reg_per_step.device, dtype=reg_per_step.dtype
        )
        base_w = 0.5 * decay_w + 0.5 * endpoint_w
        weights = base_w.unsqueeze(0).expand(bs, -1).clone()

        gt_all_xy = data["agent"]["target"][:bs, :, -t_steps:, :2]
        ego_xy = gt_all_xy[:, 0]
        other_xy = gt_all_xy[:, 1:]

        if other_xy.numel() > 0:
            dist = torch.norm(other_xy - ego_xy.unsqueeze(1), dim=-1)
            inter_any = (dist < self.interaction_distance_threshold).any(dim=1)
            for b in range(bs):
                idx = torch.nonzero(inter_any[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                t_in = int(idx[0].item())
                t_out = int(idx[-1].item())
                weights[b, t_in : t_out + 1] = 1.0

        weighted_mask = weights * valid_mask.float()
        reg_loss = (reg_per_step * weighted_mask).sum() / (
            weighted_mask.sum() + 1e-6
        )

        endpoint_loss = F.smooth_l1_loss(
            best_trajectory[:, -1, :2],
            target[:, -1, :2],
            reduction="mean",
        )
        reg_loss = reg_loss + 0.5 * endpoint_loss

        mode_endpoints = traj_modes[:, :, -1, :2]
        gt_endpoint = target[:, -1, :2]
        dist_sq = ((mode_endpoints - gt_endpoint.unsqueeze(1)) ** 2).sum(-1)
        gaussian_log_prob = -dist_sq / (2.0 * (self.gmm_sigma ** 2))
        log_pi = F.log_softmax(prob_logits, dim=-1)
        cls_loss = -torch.logsumexp(log_pi + gaussian_log_prob, dim=-1).mean()

        pairwise_dist = torch.cdist(mode_endpoints, mode_endpoints)
        too_close = F.relu(self.diversity_min_sep - pairwise_dist)
        eye = torch.eye(ka, device=too_close.device, dtype=too_close.dtype).unsqueeze(0)
        too_close = too_close * (1.0 - eye)
        diversity_loss = too_close.mean()

        return reg_loss, cls_loss, collision_loss, diversity_loss

    def _compute_metrics(self, res, data, prefix):
        trajectory = res["trajectory"]
        probability = res["probability"]

        bs, r_dim, m_dim, t_steps, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, r_dim * m_dim, t_steps, -1)
        probability = probability.reshape(bs, r_dim * m_dim)

        k = min(6, probability.shape[1])
        top_k_prob, top_k_index = probability.topk(k, dim=-1)
        top_k_traj = trajectory[
            torch.arange(bs, device=trajectory.device)[:, None], top_k_index
        ]

        if self.global_step == 0:
            ego_target_diag = data["agent"]["position"][:, 0, self.history_steps:]
            pred_target_diag = data["agent"]["position"][:, 1:, self.history_steps:]

            print(
                f"[DIAG] ego target xy range:  "
                f"{ego_target_diag[..., :2].abs().max().item():.3f}  ← should be <10"
            )
            print(
                f"[DIAG] pred target xy range: "
                f"{pred_target_diag[..., :2].abs().max().item():.3f}  ← should be <10"
            )
            print(
                f"[DIAG] traj xy range:        "
                f"{top_k_traj[..., :2].abs().max().item():.3f}   ← should be <10"
            )
            print(
                f"[DIAG] prediction xy range:  "
                f"{res['prediction'][..., :2].abs().max().item():.3f}  ← should be <10"
            )

            traj_xy = trajectory[..., :2]
            ego_target_xy_diag = ego_target_diag[..., :2]
            endpoint_dist = torch.norm(
                traj_xy[:, :, -1, :] - ego_target_xy_diag[:, -1:, :], dim=-1
            )
            print(
                f"[DIAG] min endpoint dist: "
                f"{endpoint_dist.min(dim=-1)[0].mean().item():.3f}  ← should be <3"
            )
            print(f"[DIAG] mean endpoint dist: {endpoint_dist.mean().item():.3f}")

            pred_xy = res["prediction"][..., :2]
            pred_target_xy_diag = pred_target_diag[..., :2]
            valid = data["agent"]["valid_mask"][:, 1:, self.history_steps:].float()
            pred_dist = torch.norm(pred_xy - pred_target_xy_diag, dim=-1)
            pred_dist_valid = (pred_dist * valid).sum() / (valid.sum() + 1e-6)
            print(
                f"[DIAG] mean pred displacement: {pred_dist_valid.item():.3f}  ← should be <3"
            )

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps:],
        }
        target = data["agent"]["target"][:, 0]

        return self.metrics[prefix](outputs, target)

    def _log_step(self, loss, objectives, metrics, prefix):
        self.log(
            f"loss/{prefix}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(prefix == "train"),
        )

        for k, v in objectives.items():
            self.log(
                f"objectives/{prefix}_{k}",
                v,
                on_step=True,
                on_epoch=True,
            )

        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=(prefix == "val"),
        )

    def configure_optimizers(self):
        scale_params = [
            p for n, p in self.model.named_parameters() if "score_attn_scale" in n
        ]
        other_params = [
            p for n, p in self.model.named_parameters() if "score_attn_scale" not in n
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
