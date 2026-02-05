# +++ ADD: Hydra auxiliary supervision (Level 1) +++
hydra_cfg = getattr(cfg, "hydra_heads", None)
self.enable_hydra_aux = bool(hydra_cfg is not None and getattr(hydra_cfg, "enabled", True))

if self.enable_hydra_aux:
    # Rule-based teachers (Level 1: no map / no ESDF)
    teacher_cfg = RuleTeacherConfig(
        # If your traj[...,0:2] is not x,y, change these in config later
        xy_indices=getattr(hydra_cfg, "xy_indices", (0, 1)),
        dt=float(getattr(hydra_cfg, "dt", 0.5)),
        max_step_distance=getattr(hydra_cfg, "max_step_distance", 15.0),
        max_speed=getattr(hydra_cfg, "max_speed", None),
        speed_index=getattr(hydra_cfg, "speed_index", None),

        w_collision=float(getattr(hydra_cfg, "w_collision", 0.0)),  # Level 1 default = 0
        w_offroad=float(getattr(hydra_cfg, "w_offroad", 0.0)),      # Level 1 default = 0
        w_comfort=float(getattr(hydra_cfg, "w_comfort", 1.0)),
        w_progress=float(getattr(hydra_cfg, "w_progress", -5.0)),

        soft_feasibility=bool(getattr(hydra_cfg, "soft_feasibility", True)),
        feasibility_softness=float(getattr(hydra_cfg, "feasibility_softness", 10.0)),
    )
    self.rule_teachers = RuleBasedTeachers(teacher_cfg)

    # Hydra losses
    loss_cfg = HydraLossConfig(
        w_feasibility=float(getattr(hydra_cfg, "loss_w_feasibility", 0.05)),
        w_cost=float(getattr(hydra_cfg, "loss_w_cost", 0.10)),
        w_progress=float(getattr(hydra_cfg, "loss_w_progress", 0.02)),
        w_comfort=float(getattr(hydra_cfg, "loss_w_comfort", 0.02)),
        w_rank=float(getattr(hydra_cfg, "loss_w_rank", 0.0)),

        use_focal_for_feas=bool(getattr(hydra_cfg, "use_focal_for_feas", False)),
        focal_gamma=float(getattr(hydra_cfg, "focal_gamma", 2.0)),
        focal_alpha=float(getattr(hydra_cfg, "focal_alpha", 0.25)),

        huber_delta=float(getattr(hydra_cfg, "huber_delta", 1.0)),
        mask_regression_by_feasibility=bool(getattr(hydra_cfg, "mask_regression_by_feasibility", True)),
        feas_mask_threshold=float(getattr(hydra_cfg, "feas_mask_threshold", 0.5)),

        enable_cost_uncertainty_nll=bool(getattr(hydra_cfg, "enable_cost_uncertainty_nll", False)),
    )
    self.hydra_losses = HydraLosses(loss_cfg)
else:
    self.rule_teachers = None
    self.hydra_losses = None
# --- END ---




# +++ ADD: Hydra auxiliary losses (Level 1) +++
if self.enable_hydra_aux:
    hydra_out = out.get("hydra_heads", None) if isinstance(out, dict) else None
    traj = out["trajectory"] if isinstance(out, dict) else None

    # Only compute if model actually returned head outputs
    if (hydra_out is not None) and (traj is not None) and (self.rule_teachers is not None) and (self.hydra_losses is not None):
        teacher = self.rule_teachers(traj, context=None)  # Level 1: no map/context
        aux_losses = self.hydra_losses(hydra_out, teacher)

        # Add to total loss
        loss = loss + aux_losses["loss_hydra_total"]

        # Logging (step + epoch)
        self.log("train/hydra_total", aux_losses["loss_hydra_total"], on_step=True, on_epoch=True, prog_bar=False)
        for k, v in aux_losses.items():
            if k == "loss_hydra_total":
                continue
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=False)

        # Optional: quick teacher stats (useful for debugging)
        self.log("train/teacher_feasible_rate", (teacher["feasibility"] >= 0.5).float().mean(), on_step=True, on_epoch=True, prog_bar=False)
# --- END ---





# +++ ADD in validation_step after base val loss computed +++
if self.enable_hydra_aux:
    hydra_out = out.get("hydra_heads", None)
    traj = out.get("trajectory", None)
    if (hydra_out is not None) and (traj is not None) and (self.rule_teachers is not None) and (self.hydra_losses is not None):
        teacher = self.rule_teachers(traj, context=None)
        aux_losses = self.hydra_losses(hydra_out, teacher)
        val_loss = val_loss + aux_losses["loss_hydra_total"]

        self.log("val/hydra_total", aux_losses["loss_hydra_total"], on_step=False, on_epoch=True, prog_bar=False)
        for k, v in aux_losses.items():
            if k == "loss_hydra_total":
                continue
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
# --- END ---
