weights_topk = [1.0, 0.25, 0.1]
reg_loss = 0.0

for rank in range(topk_idx.shape[1]):
    idx = topk_idx[:, rank]

    ranked_traj = traj_modes[
        torch.arange(bs, device=traj_modes.device),
        idx
    ]

    ranked_reg = F.smooth_l1_loss(
        ranked_traj,
        target,
        reduction="none"
    ).sum(-1)

    time_idx = torch.arange(
        T,
        device=ranked_reg.device,
        dtype=ranked_reg.dtype
    )

    decay_w = torch.exp(-self.k_R * time_idx)
    endpoint_w = torch.linspace(0.5, 1.5, T, device=ranked_reg.device)
    base_w = 0.5 * decay_w + 0.5 * endpoint_w
    rank_weights = base_w.unsqueeze(0).expand(bs, -1)

    weighted_mask = rank_weights * valid_mask.float()

    ranked_reg_loss = (
        ranked_reg * weighted_mask
    ).sum() / (weighted_mask.sum() + 1e-6)

    reg_loss = reg_loss + weights_topk[rank] * ranked_reg_loss
