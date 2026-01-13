    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, targets, scenarios = batch
        data = features["feature"].data

        # -------------------------------------------------
        # Debug: check INPUT data for NaN/Inf
        # -------------------------------------------------
        def _check_finite(x, path=""):
            if isinstance(x, torch.Tensor):
                if not torch.isfinite(x).all():
                    bad_token = getattr(scenarios[0], "token", "UNKNOWN") if len(scenarios) > 0 else "UNKNOWN"
                    raise RuntimeError(
                        f"[NaN debug] NaN/Inf in INPUT at '{path}' "
                        f"for scenario {bad_token}"
                    )
            elif isinstance(x, dict):
                for k, v in x.items():
                    _check_finite(v, f"{path}.{k}" if path else k)
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    _check_finite(v, f"{path}[{i}]")
            # other types (str, int, None, etc.) are ignored

        _check_finite(data, path="feature")

        # Forward
        res = self.forward(data)

        # -------------------------------------------------
        # Debug: check model outputs for NaN/Inf
        # -------------------------------------------------
        for name, tensor in res.items():
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                bad_token = getattr(scenarios[0], "token", "UNKNOWN") if len(scenarios) > 0 else "UNKNOWN"
                raise RuntimeError(
                    f"[NaN debug] NaN/Inf in model output '{name}' "
                    f"for scenario {bad_token}"
                )

        # Compute losses
        losses = self._compute_objectives(res, data)

        # Debug: check loss components
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                if not torch.isfinite(val).all():
                    bad_token = getattr(scenarios[0], "token", "UNKNOWN") if len(scenarios) > 0 else "UNKNOWN"
                    raise RuntimeError(
                        f"[NaN debug] NaN/Inf in loss component '{name}' "
                        f"for scenario {bad_token}"
                    )

        metrics = self._compute_metrics(res, data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        if self.training and (not torch.isfinite(losses["loss"])):
            bad_token = getattr(scenarios[0], "token", "UNKNOWN") if len(scenarios) > 0 else "UNKNOWN"
            raise RuntimeError(
                f"[NaN debug] Total loss is NaN/Inf for scenario {bad_token}"
            )

        return losses["loss"] if self.training else torch.tensor(0.0, device=self.device)
