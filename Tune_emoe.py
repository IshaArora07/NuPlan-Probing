import optuna
import subprocess
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─── EDIT THESE ───────────────────────────────────────────────────────────────
CACHE_PATH      = "/nuplan/exp/cache_pluto_1M"  # ← your actual cache path
TRAINING_CONFIG = "train_pluto"                  # ← your +training= config name
                                                 #   (e.g. train_emoe if renamed)
VAL_METRIC_TAG  = "loss/val/total_loss"          # ← update after Step 1 below
GPU_ID          = "0"                            # ← your GPU
MAX_WORKERS     = 4                              # ← your CPU workers
BATCH_SIZE      = 4
NUM_WORKERS     = 1
# ──────────────────────────────────────────────────────────────────────────────

N_TRIALS   = 15
OUTPUT_DIR = Path("optuna_trials")
OUTPUT_DIR.mkdir(exist_ok=True)


def objective(trial: optuna.Trial) -> float:

    # ── Sample the 3 hyperparameters ──────────────────────────────────────
    lr            = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)

    trial_dir = OUTPUT_DIR / f"trial_{trial.number}"
    trial_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Trial {trial.number:>3} | lr={lr:.2e} | wd={weight_decay:.2e} | warmup={warmup_epochs}")
    print(f"{'='*60}")

    # ── Command — mirrors PLUTO's README exactly ───────────────────────────
    cmd = [
        f"CUDA_VISIBLE_DEVICES={GPU_ID}",
        "python", "run_training.py",
        "py_func=train",
        f"+training={TRAINING_CONFIG}",
        "worker=single_machine_thread_pool",
        f"worker.max_workers={MAX_WORKERS}",
        "scenario_builder=nuplan",
        f"cache.cache_path={CACHE_PATH}",
        "cache.use_cache_without_dataset=true",
        f"data_loader.params.batch_size={BATCH_SIZE}",
        f"data_loader.params.num_workers={NUM_WORKERS}",
        "epochs=1",
        f"lr={lr}",
        f"weight_decay={weight_decay}",
        f"warmup_epochs={warmup_epochs}",
        "wandb.mode=disabled",                    # disable wandb during tuning
        f"hydra.run.dir={trial_dir}",             # redirect all outputs here
    ]

    # CUDA_VISIBLE_DEVICES must be set as env var, not inline
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    result = subprocess.run(
        # drop the first element (CUDA_VISIBLE_DEVICES=X) from cmd list
        cmd[1:],
        capture_output=True,
        text=True,
        cwd=".",
        env=env,
    )

    if result.returncode != 0:
        print(f"  ✗ Training failed:")
        print(result.stderr[-2000:])
        raise optuna.exceptions.TrialPruned()

    # ── Read from TensorBoard events file ─────────────────────────────────
    val_metric = _read_tensorboard_metric(trial_dir)
    print(f"  ✓ {VAL_METRIC_TAG} = {val_metric:.6f}")
    return val_metric


def _read_tensorboard_metric(trial_dir: Path) -> float:
    event_files = list(trial_dir.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"  ⚠ No TensorBoard events file found under {trial_dir}")
        print(f"     Contents: {list(trial_dir.rglob('*'))}")
        return float("inf")

    event_path = event_files[0].parent
    print(f"  📄 Reading: {event_path}")

    ea = EventAccumulator(str(event_path))
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    print(f"     Tags available: {available_tags}")   # printed every trial

    if VAL_METRIC_TAG not in available_tags:
        print(f"  ⚠ Tag '{VAL_METRIC_TAG}' not found!")
        print(f"     Available: {available_tags}")
        return float("inf")

    events = ea.Scalars(VAL_METRIC_TAG)
    if not events:
        return float("inf")

    return float(events[-1].value)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="emoe_lr_wd_warmup",
        direction="minimize",
        storage=f"sqlite:///{OUTPUT_DIR}/optuna.db",
        load_if_exists=True,    # safe to resume if interrupted
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 60)
    print(f"  Best trial  : #{study.best_trial.number}")
    print(f"  Best value  : {study.best_value:.6f}")
    print(f"  Best params :")
    for k, v in study.best_params.items():
        print(f"    {k:20s} = {v}")

    best_path = OUTPUT_DIR / "best_params.json"
    best_path.write_text(
        json.dumps({"value": study.best_value, "params": study.best_params}, indent=2)
    )
    print(f"\n  Saved → {best_path}")
