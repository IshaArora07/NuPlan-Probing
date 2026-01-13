    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, targets, scenarios = batch
        data = features["feature"].data

        # ------------------------------------------------------------------
        # DEBUG: identify scenarios in this batch
        # ------------------------------------------------------------------
        scenario_tokens = [getattr(s, "token", "UNKNOWN") for s in scenarios]

        # ------------------------------------------------------------------
        # 0) Check inputs for NaN / Inf
        # ------------------------------------------------------------------
        def _check_tensor(t, name: str):
            if not torch.isfinite(t).all():
                raise RuntimeError(
                    f"[NaN debug] non-finite values in INPUT tensor '{name}' "
                    f"for scenarios {scenario_tokens}"
                )

        try:
            _check_tensor(data["agent"]["target"],           "agent.target")
            _check_tensor(data["agent"]["valid_mask"],       "agent.valid_mask")
            _check_tensor(data["agent"]["velocity"],         "agent.velocity")
            _check_tensor(data["reference_line"]["future_projection"],
                          "reference_line.future_projection")
            _check_tensor(data["cost_maps"],                 "cost_maps")
        except KeyError:
            # if any of these keys do not exist we just skip the check
            pass

        # ------------------------------------------------------------------
        # 1) Forward pass
        # ------------------------------------------------------------------
        res = self.forward(data)

        # ------------------------------------------------------------------
        # 2) Check model outputs for NaN / Inf and sanitise
        # ------------------------------------------------------------------
        if "prediction" in res:
            pred = res["prediction"]
            finite_mask = torch.isfinite(pred)

            if not finite_mask.all():
                # Which batch elements are bad?
                bs = pred.shape[0]
                bad_batch = (~torch.isfinite(pred.view(bs, -1)).all(dim=1)).nonzero(as_tuple=False).flatten()
                bad_tokens = [scenario_tokens[i] for i in bad_batch.tolist()]

                logger.error(
                    "[NaN debug] NaN/Inf in model output 'prediction' for "
                    f"{len(bad_tokens)} scenario(s): {bad_tokens}"
                )

                # Sanitise: replace non-finite with zeros so loss stays finite
                pred = torch.where(finite_mask, pred, torch.zeros_like(pred))
                res["prediction"] = pred

        # ------------------------------------------------------------------
        # 3) Normal loss + metrics
        # ------------------------------------------------------------------
        losses = self._compute_objectives(res, data)
        metrics = self._compute_metrics(res, data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0








    # ----------------------------------------------------------------------
    # NEW: helper to log tokens that caused non-finite loss
    # ----------------------------------------------------------------------
    def _log_bad_tokens(self, scenario_tokens, reason: str, prefix: str) -> None:
        """
        Append bad scenario tokens to a text file and log via logger.
        """
        if not scenario_tokens:
            return

        # Update in-memory set
        for t in scenario_tokens:
            if t is not None:
                self.bad_tokens.add(str(t))

        # Log to file
        try:
            with open(self.bad_token_log_path, "a") as f:
                for t in scenario_tokens:
                    if t is None:
                        continue
                    f.write(f"{prefix}\t{reason}\t{str(t)}\n")
        except Exception as e:
            logger.error(f"Failed to write bad tokens to {self.bad_token_log_path}: {e}")

        # Also log a summary line
        logger.warning(
            "[LightningTrainer] Skipping batch (%s) due to %s; tokens: %s",
            prefix,
            reason,
            ", ".join(str(t) for t in scenario_tokens if t is not None),
        )







    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, targets, scenarios = batch

        # ------------------------------------------------------------------
        # NEW: Early skip if this batch contains tokens already known as bad
        # ------------------------------------------------------------------
        scenario_tokens = []
        for sc in scenarios:
            tok = getattr(sc, "token", None)
            scenario_tokens.append(tok)

        # If you want to completely skip any batch that *contains* a known
        # bad token, keep this; otherwise you can remove this block.
        if any((t is not None and str(t) in self.bad_tokens) for t in scenario_tokens):
            self._log_bad_tokens(scenario_tokens, reason="skip_known_bad", prefix=prefix)
            # Return a zero loss tensor that still has grad so Lightning is happy
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # ------------------------------------------------------------------
        # Normal forward + loss
        # ------------------------------------------------------------------
        data = features["feature"].data
        res = self.forward(data)

        # Compute objectives first
        losses = self._compute_objectives(res, data)

        total_loss = losses["loss"]

        # ------------------------------------------------------------------
        # NEW: detect non-finite loss and skip batch
        # ------------------------------------------------------------------
        if not torch.isfinite(total_loss):
            # Log and remember these tokens
            self._log_bad_tokens(scenario_tokens, reason="nan_or_inf_loss", prefix=prefix)

            # Return a zero loss whose backward is well-defined
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Only compute metrics if loss is finite
        metrics = self._compute_metrics(res, data, prefix)
        self._log_step(total_loss, losses, metrics, prefix)

        return total_loss if self.training else torch.tensor(0.0, device=self.device)


