# +++ ADD: Hydra auxiliary supervision (Level 1) +++
# Enable via hyperparameter; default off unless explicitly passed
self.use_hydra_aux = bool(getattr(self.hparams, "use_hydra_aux", False))

if self.use_hydra_aux:
    # Rule-based teachers (Level 1: no ESDF/map usage here)
    teacher_cfg = RuleTeacherConfig(
        xy_indices=getattr(self.hparams, "hydra_xy_indices", (0, 1)),
        dt=float(getattr(self.hparams, "hydra_dt", 0.5)),
        max_step_distance=getattr(self.hparams, "hydra_max_step_distance", 15.0),
        max_speed=getattr(self.hparams, "hydra_max_speed", None),
        speed_index=getattr(self.hparams, "hydra_speed_index", None),

        # Level 1: collision/offroad terms disabled by default
        w_collision=float(getattr(self.hparams, "hydra_w_collision", 0.0)),
        w_offroad=float(getattr(self.hparams, "hydra_w_offroad", 0.0)),
        w_comfort=float(getattr(self.hparams, "hydra_w_comfort", 1.0)),
        w_progress=float(getattr(self.hparams, "hydra_w_progress", -5.0)),

        soft_feasibility=bool(getattr(self.hparams, "hydra_soft_feasibility", True)),
        feasibility_softness=float(getattr(self.hparams, "hydra_feasibility_softness", 10.0)),
    )
    self.rule_teachers = RuleBasedTeachers(teacher_cfg)

    loss_cfg = HydraLossConfig(
        w_feasibility=float(getattr(self.hparams, "hydra_loss_w_feasibility", 0.05)),
        w_cost=float(getattr(self.hparams, "hydra_loss_w_cost", 0.10)),
        w_progress=float(getattr(self.hparams, "hydra_loss_w_progress", 0.02)),
        w_comfort=float(getattr(self.hparams, "hydra_loss_w_comfort", 0.02)),
        w_rank=float(getattr(self.hparams, "hydra_loss_w_rank", 0.0)),

        huber_delta=float(getattr(self.hparams, "hydra_huber_delta", 1.0)),
        mask_regression_by_feasibility=bool(getattr(self.hparams, "hydra_mask_regression_by_feasibility", True)),
        feas_mask_threshold=float(getattr(self.hparams, "hydra_feas_mask_threshold", 0.5)),
    )
    self.hydra_losses = HydraLosses(loss_cfg)
else:
    self.rule_teachers = None
    self.hydra_losses = None
# --- END ---





# +++ ADD: Hydra auxiliary heads (Level 1) +++
hydra_total = ego_reg_loss.new_zeros(())  # scalar tensor on correct device
hydra_losses_dict = {}

if self.use_hydra_aux and (self.rule_teachers is not None) and (self.hydra_losses is not None):
    hydra_out = res.get("hydra_heads", None)

    # Only compute if model returned head outputs
    if hydra_out is not None:
        # trajectory is (bs, 1, Ka, T, 6) -> use ego reference line index 0 => (bs, Ka, T, 6)
        traj_all_modes = trajectory[:, 0]  # already sliced to train_num above

        teacher = self.rule_teachers(traj_all_modes, context=None)
        aux = self.hydra_losses(hydra_out, teacher)

        hydra_total = aux["loss_hydra_total"]
        loss = loss + hydra_total

        # Keep individual aux losses for logging via objectives
        hydra_losses_dict = aux
# --- END ---





# +++ ADD to returned objectives dict +++
"hydra_total": float(hydra_total.detach().item()),
"hydra_loss_feasibility": float(hydra_losses_dict.get("loss_feasibility", hydra_total.new_zeros(())).detach().item()),
"hydra_loss_cost": float(hydra_losses_dict.get("loss_cost", hydra_total.new_zeros(())).detach().item()),
"hydra_loss_progress": float(hydra_losses_dict.get("loss_progress", hydra_total.new_zeros(())).detach().item()),
"hydra_loss_comfort": float(hydra_losses_dict.get("loss_comfort", hydra_total.new_zeros(())).detach().item()),
"teacher_feasible_rate": float(
    (teacher["feasibility"] >= 0.5).float().mean().detach().item()
) if (self.use_hydra_aux and (self.rule_teachers is not None) and ("teacher" in locals())) else 0.0,
# --- END ---





