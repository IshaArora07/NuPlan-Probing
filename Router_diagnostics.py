# ================= ROUTER DIAGNOSTICS =================
if router_logits is not None and "emoe" in data:
    router_probs = F.softmax(router_logits[:train_num], dim=-1)  # [B, S]
    router_choice = router_probs.argmax(dim=-1)                  # [B]

    with torch.no_grad():
        # Expert usage histogram
        usage = torch.bincount(
            router_choice,
            minlength=router_probs.shape[-1]
        ).float()

        usage_frac = usage / usage.sum().clamp(min=1)

        # Mean max-probability (confidence)
        max_prob = router_probs.max(dim=-1).values.mean()

        # Entropy (collapse detector)
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()

    # Log global router stats
    self.log("router/max_prob", max_prob, on_epoch=True, prog_bar=False)
    self.log("router/entropy", entropy, on_epoch=True, prog_bar=False)

    for i, u in enumerate(usage_frac):
        self.log(
            f"router/usage_frac_expert_{i}",
            u,
            on_epoch=True,
            prog_bar=False,
        )
# =====================================================
