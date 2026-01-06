    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory:  (bs, R_model, M=Ka, T, 6)
        probability: (bs, R_model, M=Ka)
        valid_mask:  (bs, T)       - ego future valid mask
        target:      (bs, T, 6)    - GT ego trajectory in local frame
        """
        # ----- NEW: align route dimension with reference_line -----
        # reference_line has shape (bs, R_ref, N_points)
        R_ref = data["reference_line"]["valid_mask"].shape[1]
        R_model = probability.shape[1]

        if R_model == 1 and R_ref > 1:
            # tile model outputs along route dimension
            probability = probability.expand(bs, R_ref, -1)          # (bs, R_ref, Ka)
            trajectory = trajectory.expand(bs, R_ref, -1, -1, -1)    # (bs, R_ref, Ka, T, 6)
        elif R_model != R_ref:
            # hard error in any other mismatch case so we notice
            raise ValueError(
                f"Route dimension mismatch: probability has R={R_model}, "
                f"reference_line has R={R_ref}."
            )
        # ----- END NEW -----

        # same reference-line based mode selection as original PLUTO
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)   # (bs, R_ref)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        ...








    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        trajectory, probability = res["trajectory"], res["probability"]

        # ----- NEW: align route dimension with reference_line -----
        bs = trajectory.shape[0]
        R_ref = data["reference_line"]["valid_mask"].shape[1]
        R_model = probability.shape[1]

        if R_model == 1 and R_ref > 1:
            probability = probability.expand(bs, R_ref, -1)          # (bs, R_ref, Ka)
            trajectory = trajectory.expand(bs, R_ref, -1, -1, -1)    # (bs, R_ref, Ka, T, 6)
        elif R_model != R_ref:
            raise ValueError(
                f"Route dimension mismatch in metrics: R_model={R_model}, R_ref={R_ref}"
            )
        # ----- END NEW -----

        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)  # (bs, R_ref)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        ...

