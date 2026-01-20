# ---------------------------
# Add to your LightningTrainer
# ---------------------------
import torch
import torch.distributed as dist


class LightningTrainer(pl.LightningModule):
    def __init__(self, ..., num_scene_types: int = 6, router_cm_bins=None, **kwargs):
        super().__init__()
        ...
        self.num_scene_types = num_scene_types  # S
        # Confidence bins for Option C
        # default: [0.0-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        if router_cm_bins is None:
            router_cm_bins = [0.0, 0.4, 0.6, 0.8, 1.0]
        self.router_cm_bins = torch.tensor(router_cm_bins, dtype=torch.float32)
        self.num_bins = len(router_cm_bins) - 1

        # Will be allocated on device in on_fit_start
        self._cm_hard = None          # [BINS, S, S] int64 (counts)
        self._cm_soft = None          # [BINS, S, S] float32 (expected counts)
        self._cm_bin_counts = None    # [BINS] int64 (how many samples in bin)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        ...
        # Allocate confusion-matrix accumulators
        S = self.num_scene_types
        B = self.num_bins
        device = self.device

        self._cm_hard = torch.zeros((B, S, S), dtype=torch.long, device=device)
        self._cm_soft = torch.zeros((B, S, S), dtype=torch.float32, device=device)
        self._cm_bin_counts = torch.zeros((B,), dtype=torch.long, device=device)

        # Keep bins tensor on device too
        self.router_cm_bins = self.router_cm_bins.to(device)

    # ---------------------------
    # Reset per validation epoch
    # ---------------------------
    def on_validation_epoch_start(self) -> None:
        if self._cm_hard is not None:
            self._cm_hard.zero_()
            self._cm_soft.zero_()
            self._cm_bin_counts.zero_()

    # ---------------------------
    # Accumulator (Option C)
    # ---------------------------
    @torch.no_grad()
    def _accumulate_router_confusion(self, res, data) -> None:
        """
        Builds Option C:
          - Hard confusion (argmax expert vs true class)
          - Soft confusion (expected counts using router probabilities)
          - Stratified by confidence bins (max prob)
        Matrix convention: rows = routed expert k, cols = true class y
        """
        if res is None:
            return
        if "router_probs" not in res:
            return
        if "emoe" not in data or not isinstance(data["emoe"], dict) or "scene_label" not in data["emoe"]:
            return

        probs = res["router_probs"]                       # [bs, S]
        y = data["emoe"]["scene_label"].long()            # [bs]
        bs, S = probs.shape
        if S != self.num_scene_types:
            return

        # Confidence and hard expert
        conf, k_hat = probs.max(dim=-1)                   # [bs], [bs]

        # Bin index for each sample: 0..B-1
        # bucketize returns index in [0..len(bins)] for right=False,
        # we want bin b such that bins[b] <= conf < bins[b+1]
        bin_idx = torch.bucketize(conf, self.router_cm_bins, right=False) - 1
        bin_idx = bin_idx.clamp(min=0, max=self.num_bins - 1)  # [bs]

        # Update bin counts
        self._cm_bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long))

        # HARD confusion: increment (bin, k_hat, y)
        # Build flat indices: (k_hat * S + y) in [0..S*S-1]
        flat_h = (k_hat * S + y)                          # [bs]
        for b in range(self.num_bins):
            m = (bin_idx == b)
            if not m.any():
                continue
            flat_b = flat_h[m]
            # count occurrences of each (k,y)
            counts = torch.bincount(flat_b, minlength=S * S).to(self._cm_hard.dtype)
            self._cm_hard[b].view(-1).add_(counts)

        # SOFT confusion: add probs mass into column y, per row k
        # For each sample i with label y_i, add probs[i, k] to (k, y_i).
        for b in range(self.num_bins):
            m = (bin_idx == b)
            if not m.any():
                continue
            probs_b = probs[m]    # [nb, S]
            y_b = y[m]            # [nb]

            # For each class y, sum probs of samples with that y
            # This yields a matrix [S, S] where column y is sum over samples of that class
            for cls in range(S):
                mc = (y_b == cls)
                if not mc.any():
                    continue
                self._cm_soft[b, :, cls] += probs_b[mc].sum(dim=0)

    # ---------------------------
    # Hook accumulation into validation
    # ---------------------------
    def validation_step(self, batch, batch_idx: int):
        features, targets, scenarios = batch
        data = features["feature"].data
        res = self.forward(data)

        # accumulate Option C router confusion
        self._accumulate_router_confusion(res, data)

        losses = self._compute_objectives(res, data)
        metrics = self._compute_metrics(res, data, "val")
        self._log_step(losses["loss"], losses, metrics, "val")
        return losses["loss"]

    # ---------------------------
    # Reduce across DDP and print/log
    # ---------------------------
    def on_validation_epoch_end(self) -> None:
        # DDP sync
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._cm_hard, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._cm_soft, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._cm_bin_counts, op=dist.ReduceOp.SUM)

        if not self.trainer.is_global_zero:
            return

        # Pretty print summaries
        bins = self.router_cm_bins.detach().cpu().tolist()
        S = self.num_scene_types

        def _col_norm(mat):
            # mat: [S,S] (rows=k, cols=y)
            colsum = mat.sum(dim=0, keepdim=True).clamp_min(1e-9)
            return mat / colsum

        print("\n[Router Option C] Confusion matrices (rows=expert k, cols=true class y)")
        for b in range(self.num_bins):
            lo, hi = bins[b], bins[b + 1]
            n = int(self._cm_bin_counts[b].item())
            hard = self._cm_hard[b].detach().cpu()
            soft = self._cm_soft[b].detach().cpu()

            print(f"\n  Bin {b}: conf in [{lo:.1f}, {hi:.1f})   N={n}")

            # Hard, column-normalized
            hard_f = hard.to(torch.float32)
            hard_cn = _col_norm(hard_f)
            print("    HARD (argmax) column-normalized:")
            print(hard_cn.numpy())

            # Soft, column-normalized
            soft_cn = _col_norm(soft)
            print("    SOFT (expected mass) column-normalized:")
            print(soft_cn.numpy())

            # Extra: diagonal quality signals
            hard_acc_per_class = torch.diag(hard_cn)  # P(k=y | true=y) in this bin
            soft_diag_mass = torch.diag(soft_cn)      # average probability mass on correct expert per class (normalized view)
            print("    diag(hard) per class:", hard_acc_per_class.numpy())
            print("    diag(soft) per class:", soft_diag_mass.numpy())

        # Optional: log a single scalar per epoch (overall hard routing accuracy)
        hard_total = self._cm_hard.sum(dim=0).to(torch.float32)  # [S,S]
        hard_total_cn = _col_norm(hard_total)
        hard_diag_mean = torch.diag(hard_total_cn).mean()
        self.log("val/router_hard_diag_mean", hard_diag_mean, on_epoch=True, prog_bar=False, sync_dist=True)
