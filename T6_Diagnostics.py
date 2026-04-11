# ======== T6 DIAGNOSTICS ========
with torch.no_grad():
    # 1. Mode entropy — are probabilities spread?
    probs_soft = F.softmax(prob_logits, dim=-1)
    mode_entropy = -(
        probs_soft * torch.log(probs_soft + 1e-9)
    ).sum(dim=-1).mean()
    mode_max_prob = probs_soft.max(dim=-1).values.mean()

    # 2. WTA winner distribution — is WTA selecting diverse modes?
    winner_counts = torch.bincount(
        target_m_index,
        minlength=Ka
    ).float()
    winner_entropy = -(
        (winner_counts / winner_counts.sum()) *
        torch.log(winner_counts / winner_counts.sum() + 1e-9)
    ).sum()
    mode0_wins = (target_m_index == 0).float().mean()

    # 3. Trajectory endpoint diversity
    endpoints = traj_modes[:, :, -1, :2]  # [B, Ka, 2]
    traj_y_std = endpoints[:, :, 1].std(dim=-1).mean()
    traj_x_range = (
        endpoints[:, :, 0].max(dim=-1).values -
        endpoints[:, :, 0].min(dim=-1).values
    ).mean()

    # 4. Anchor spread (only if learnable)
    # Skip if frozen buffer

    # Log everything
    self.log("diag/mode_entropy", 
             mode_entropy, on_step=True)
    self.log("diag/mode_max_prob", 
             mode_max_prob, on_step=True)
    self.log("diag/wta_winner_entropy", 
             winner_entropy, on_step=True)
    self.log("diag/wta_mode0_wins", 
             mode0_wins, on_step=True)
    self.log("diag/traj_endpoint_y_std", 
             traj_y_std, on_step=True)
    self.log("diag/traj_endpoint_x_range", 
             traj_x_range, on_step=True)
    self.log("diag/diversity_loss", 
             diversity_loss, on_step=True)
    self.log("diag/all_modes_reg", 
             all_modes_reg, on_step=True)
# ======== END DIAGNOSTICS ========
