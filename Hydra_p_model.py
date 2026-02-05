# +++ ADD: Hydra heads (auxiliary) +++
# Enable flag (default True if config missing; adjust to your config conventions)
self.enable_hydra_heads = bool(getattr(cfg, "hydra_heads", None) and getattr(cfg.hydra_heads, "enabled", True))

if self.enable_hydra_heads:
    # decoded_queries is [B, Ka, d_model], so in_dim = d_model
    hydra_cfg = HydraHeadsConfig(
        in_dim=self.d_model,  # or whatever your decoder query dim is called
        trunk_hidden_dim=getattr(cfg.hydra_heads, "trunk_hidden_dim", 256),
        trunk_depth=getattr(cfg.hydra_heads, "trunk_depth", 2),
        trunk_dropout=getattr(cfg.hydra_heads, "trunk_dropout", 0.0),
        use_layernorm=getattr(cfg.hydra_heads, "use_layernorm", True),

        enable_feasibility=getattr(cfg.hydra_heads, "enable_feasibility", True),
        enable_cost=getattr(cfg.hydra_heads, "enable_cost", True),
        enable_progress=getattr(cfg.hydra_heads, "enable_progress", True),
        enable_comfort=getattr(cfg.hydra_heads, "enable_comfort", False),
        enable_uncertainty=getattr(cfg.hydra_heads, "enable_uncertainty", False),

        head_hidden_dim=getattr(cfg.hydra_heads, "head_hidden_dim", 128),
        head_depth=getattr(cfg.hydra_heads, "head_depth", 2),
        head_dropout=getattr(cfg.hydra_heads, "head_dropout", 0.0),

        # keep 0 for now (Level 1): no router conditioning
        router_probs_dim=0,
    )
    self.hydra_heads = HydraPredictionHeads(hydra_cfg)
else:
    self.hydra_heads = None
# --- END ---



