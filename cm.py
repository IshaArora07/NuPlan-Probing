# ============================================================
# router_confusion_callback.py
# Option C: Hard + Soft confusion matrices + confidence bins
# ============================================================
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import pytorch_lightning as pl

try:
    import torch.distributed as dist
except Exception:
    dist = None


@dataclass
class RouterConfusionConfig:
    num_scene_types: int = 6                 # S
    conf_bin_edges: Tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.0000001)  # 4 bins
    save_dir: str = "router_confusion"       # relative to logger dir if available
    run_on: str = "val"                      # "val" or "train"
    normalize: str = "col"                   # "none" | "col" | "row"
    log_to_tensorboard: bool = False         # optional image logging (requires TB logger)


def _is_dist() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _all_reduce_(x: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _normalize_matrix(mat: torch.Tensor, mode: str) -> torch.Tensor:
    # mat: [K, Y]
    if mode == "none":
        return mat
    if mode == "col":
        denom = mat.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return mat / denom
    if mode == "row":
        denom = mat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return mat / denom
    raise ValueError(f"Unknown normalize mode: {mode}")


class RouterConfusionCallback(pl.Callback):
    """
    Collects routing statistics:
      - Hard confusion: count routed expert (argmax) vs true class
      - Soft confusion: sum of router_probs mass vs true class
    Stratified by confidence bins: conf = max(router_probs)
    """

    def __init__(self, cfg: RouterConfusionConfig):
        super().__init__()
        self.cfg = cfg
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        S = self.cfg.num_scene_types
        B = len(self.cfg.conf_bin_edges) - 1
        self.hard = None  # [B, S, S]
        self.soft = None  # [B, S, S]
        self.bin_counts = None  # [B]
        self.total = 0
        self.B = B
        self.S = S

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = pl_module.device
        self.hard = torch.zeros((self.B, self.S, self.S), device=device, dtype=torch.float32)
        self.soft = torch.zeros((self.B, self.S, self.S), device=device, dtype=torch.float32)
        self.bin_counts = torch.zeros((self.B,), device=device, dtype=torch.float32)
        self.total = 0

    @torch.no_grad()
    def _update(self, router_probs: torch.Tensor, scene_label: torch.Tensor) -> None:
        """
        router_probs: [B, S] float (probabilities)
        scene_label:  [B] long (true class 0..S-1)
        """
        # Safety
        router_probs = router_probs.float()
        scene_label = scene_label.long()

        # Confidence + argmax expert
        conf, routed = router_probs.max(dim=1)  # [B], [B]
        edges = torch.tensor(self.cfg.conf_bin_edges, device=router_probs.device, dtype=conf.dtype)

        # Bin index in [0, B-1]
        # bucketize returns index in [0..len(edges)]
        bin_idx = torch.bucketize(conf, edges, right=False) - 1
        bin_idx = bin_idx.clamp_min(0).clamp_max(self.B - 1)

        # Hard counts: C[bin, routed, true] += 1
        # Soft counts: C[bin, k, true] += p_k
        for b in range(self.B):
            m = (bin_idx == b)
            if not m.any():
                continue
            y = scene_label[m]             # [Nb]
            k = routed[m]                  # [Nb]
            p = router_probs[m]            # [Nb, S]

            # hard: scatter-add into [S,S] using (k,y)
            hard_flat = torch.zeros((self.S * self.S,), device=router_probs.device, dtype=torch.float32)
            idx = (k * self.S + y).view(-1)  # [Nb]
            hard_flat.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            self.hard[b] += hard_flat.view(self.S, self.S)

            # soft: for each sample i with label y_i, add p_i to column y_i
            # soft[b, :, y_i] += p_i
            soft_b = self.soft[b]  # [S,S]
            for yi in range(self.S):
                my = (y == yi)
                if my.any():
                    soft_b[:, yi] += p[my].sum(dim=0)

            self.bin_counts[b] += float(m.sum().item())
            self.total += int(m.sum().item())

    # ---------------------------
    # Hook: val or train batches
    # ---------------------------
    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Optional[Dict], batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.cfg.run_on != "val":
            return
        if outputs is None:
            return
        if "router_probs" not in outputs or "scene_label" not in outputs:
            return
        self._update(outputs["router_probs"], outputs["scene_label"])

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Optional[Dict], batch, batch_idx: int
    ) -> None:
        if self.cfg.run_on != "train":
            return
        if outputs is None:
            return
        if "router_probs" not in outputs or "scene_label" not in outputs:
            return
        self._update(outputs["router_probs"], outputs["scene_label"])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.run_on != "val":
            return
        self._finalize_and_save(trainer, pl_module, split="val")
        self._reset_epoch_only(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.run_on != "train":
            return
        self._finalize_and_save(trainer, pl_module, split="train")
        self._reset_epoch_only(trainer, pl_module)

    def _reset_epoch_only(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # reset buffers for next epoch
        device = pl_module.device
        self.hard.zero_()
        self.soft.zero_()
        self.bin_counts.zero_()
        self.total = 0

    def _finalize_and_save(self, trainer: pl.Trainer, pl_module: pl.LightningModule, split: str) -> None:
        # Distributed sync
        hard = _all_reduce_(self.hard.detach().clone())
        soft = _all_reduce_(self.soft.detach().clone())
        bin_counts = _all_reduce_(self.bin_counts.detach().clone())

        # Normalized versions
        hard_norm = torch.stack([_normalize_matrix(hard[b], self.cfg.normalize) for b in range(self.B)], dim=0)
        soft_norm = torch.stack([_normalize_matrix(soft[b], self.cfg.normalize) for b in range(self.B)], dim=0)

        # Save path
        base_dir = None
        if trainer.logger is not None and hasattr(trainer.logger, "log_dir") and trainer.logger.log_dir is not None:
            base_dir = Path(trainer.logger.log_dir)
        else:
            base_dir = Path(".")
        out_dir = (base_dir / self.cfg.save_dir / split / f"epoch_{trainer.current_epoch:03d}")
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "num_scene_types": self.S,
            "conf_bin_edges": list(self.cfg.conf_bin_edges),
            "normalize": self.cfg.normalize,
            "bin_counts": bin_counts.cpu().tolist(),
            "hard_counts": hard.cpu().tolist(),
            "soft_counts": soft.cpu().tolist(),
            "hard_norm": hard_norm.cpu().tolist(),
            "soft_norm": soft_norm.cpu().tolist(),
        }

        with (out_dir / "router_confusion.json").open("w") as f:
            json.dump(payload, f, indent=2)

        # Optional: log summary scalars
        # - Hard diagonal accuracy per bin (argmax accuracy conditioned on true label, using col-normalized)
        # - Soft diagonal mass per bin (expected probability on correct expert, using col-normalized)
        hard_col = torch.stack([_normalize_matrix(hard[b], "col") for b in range(self.B)], dim=0)  # [B,S,S]
        soft_col = torch.stack([_normalize_matrix(soft[b], "col") for b in range(self.B)], dim=0)

        diag_hard = hard_col.diagonal(dim1=1, dim2=2).mean(dim=1)  # [B] mean over classes
        diag_soft = soft_col.diagonal(dim1=1, dim2=2).mean(dim=1)  # [B]

        for b in range(self.B):
            pl_module.log(f"{split}/router_hard_diag_bin{b}", diag_hard[b], on_epoch=True, prog_bar=False, sync_dist=True)
            pl_module.log(f"{split}/router_soft_diag_bin{b}", diag_soft[b], on_epoch=True, prog_bar=False, sync_dist=True)

        pl_module.log(f"{split}/router_bin_total", bin_counts.sum(), on_epoch=True, prog_bar=False, sync_dist=True)
