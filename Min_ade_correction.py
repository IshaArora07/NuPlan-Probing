def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
    with torch.no_grad():
        pred, _ = sort_predictions(
            outputs["trajectory"], outputs["probability"], k=self.k
        )
        # Standard definition: select best mode by ENDPOINT error
        fde = torch.norm(
            pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
        )  # (B, K)
        best_mode = fde.argmin(dim=-1)  # (B,)
        best_pred = pred[torch.arange(pred.size(0), device=pred.device), best_mode]  # (B, T, 2)
        ade = torch.norm(
            best_pred[..., :2] - target[..., :2], p=2, dim=-1
        ).mean(-1)  # (B,)
        self.sum += ade.sum()
        self.count += pred.size(0)
