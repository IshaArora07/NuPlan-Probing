def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
    """
    EMoE-safe planning loss (R=1):
      - Choose best mode by WTA: argmin distance to GT (over valid steps)
      - Reg loss on that best trajectory
      - Cls loss: CE(prob_logits, best_mode_index)
      - Collision loss on best trajectory (optional)

    trajectory:  (bs, 1, Ka, T, 6)
    probability: (bs, 1, Ka)    (logits!)
    valid_mask:  (bs, T)
    target:      (bs, T, 6)
    """
    # Sanity
    assert trajectory.dim() == 5, trajectory.shape
    assert probability.dim() == 3, probability.shape
    assert trajectory.shape[0] == bs and probability.shape[0] == bs
    assert trajectory.shape[1] == 1 and probability.shape[1] == 1, (
        f"EMoE expects R=1 but got traj={trajectory.shape}, prob={probability.shape}"
    )

    Ka = trajectory.shape[2]
    T = trajectory.shape[3]

    # Flatten R=1
    traj_modes = trajectory[:, 0]      # (bs, Ka, T, 6)
    prob_logits = probability[:, 0]    # (bs, Ka)

    # Valid mask
    mask = valid_mask.float()          # (bs, T)
    mask = mask[:, None, :]            # (bs, 1, T) for broadcasting

    # -------------------------------------------------------
    # Winner-Take-All assignment: choose closest mode to GT
    # -------------------------------------------------------
    # Per-mode distance to GT (robust): SmoothL1 over 6 dims, sum over dims & time with mask
    per_step = F.smooth_l1_loss(
        traj_modes, target[:, None, :, :], reduction="none"
    ).sum(-1)                           # (bs, Ka, T)

    per_mode = (per_step * mask).sum(-1) / (mask.sum(-1) + 1e-6)  # (bs, Ka)
    target_m_index = torch.argmin(per_mode, dim=-1)               # (bs,)

    # Gather best trajectory
    best_trajectory = traj_modes[torch.arange(bs, device=traj_modes.device), target_m_index]  # (bs, T, 6)

    # -------------------------------------------------------
    # Collision loss on best trajectory (optional)
    # -------------------------------------------------------
    if self.use_collision_loss:
        collision_loss = self.collision_loss(
            best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
        )
    else:
        collision_loss = traj_modes.new_zeros(1)

    # -------------------------------------------------------
    # Regression loss (your temporal weighting stays valid)
    # -------------------------------------------------------
    reg_per_step = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)  # (bs, T)

    time_idx = torch.arange(T, device=reg_per_step.device, dtype=reg_per_step.dtype)
    base_w = torch.exp(-self.k_R * time_idx)                 # (T,)
    weights = base_w.unsqueeze(0).expand(bs, -1).clone()     # (bs, T)

    # (keep your interaction-based reweighting if you want)
    # NOTE: uses GT agent targets, same as your current code
    gt_all_xy = data["agent"]["target"][:bs, :, -T:, :2]     # (bs, A, T, 2)
    ego_xy = gt_all_xy[:, 0]                                 # (bs, T, 2)
    other_xy = gt_all_xy[:, 1:]                              # (bs, A-1, T, 2)

    if other_xy.numel() > 0:
        dist = torch.norm(other_xy - ego_xy.unsqueeze(1), dim=-1)  # (bs, A-1, T)
        inter_any = (dist < self.interaction_distance_threshold).any(dim=1)  # (bs, T)
        for b in range(bs):
            idx = torch.nonzero(inter_any[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            t_in = int(idx[0].item())
            t_out = int(idx[-1].item())
            weights[b, t_in : t_out + 1] = 1.0

    weighted_mask = weights * valid_mask.float()
    reg_loss = (reg_per_step * weighted_mask).sum() / (weighted_mask.sum() + 1e-6)

    # -------------------------------------------------------
    # Classification loss: CE over Ka modes
    # -------------------------------------------------------
    cls_loss = F.cross_entropy(prob_logits, target_m_index.detach())

    return reg_loss, cls_loss, collision_loss
