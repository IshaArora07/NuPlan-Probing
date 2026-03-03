import json
from pathlib import Path
import pytorch_lightning as pl

class MetricDumpCallback(pl.Callback):
    """Dumps Lightning callback_metrics to a JSON file at the end of each validation epoch."""
    def __init__(self, output_path: str = "metrics.json"):
        super().__init__()
        self.output_path = output_path

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v.detach().cpu().item())  # tensors
            except Exception:
                try:
                    metrics[k] = float(v)  # scalars
                except Exception:
                    pass

        out = Path(trainer.default_root_dir) / self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
