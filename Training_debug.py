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
