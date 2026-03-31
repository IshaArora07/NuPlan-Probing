#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting EMoE Training Container"

# ── PYTHONPATH ─────────────────────────────────────────────
export PYTHONPATH="/workspace/Thesis:/workspace/nuplan-devkit:${PYTHONPATH:-}"

# ── Required env vars ─────────────────────────────────────
# S3_DATA_ROOT  e.g. s3://prod-pipeline/data/nuPlan_raw
# S3_EXP_SUBDIR (optional, defaults to 'emoe')

: "${S3_DATA_ROOT:?Set S3_DATA_ROOT like s3://prod-pipeline/data/nuPlan_raw}"

S3_EXP_SUBDIR="${S3_EXP_SUBDIR:-emoe}"

# ── Training hyperparams (PLUTO-style defaults) ───────────
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_WORKERS="${MAX_WORKERS:-32}"
WARM_START_CHECKPOINT_PATH="${WARM_START_CHECKPOINT_PATH:-}"

# ── Check AWS CLI ──────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
    echo "[ERROR] aws CLI not found in container"
    exit 1
fi

# ── Local paths inside container ───────────────────────────
WORK_ROOT="/work"
LOCAL_CACHE_ROOT="${WORK_ROOT}/cache"
LOCAL_EMOE_ROOT="${WORK_ROOT}/emoe_labels"
LOCAL_EXP_ROOT="${WORK_ROOT}/experiments/${S3_EXP_SUBDIR}"

mkdir -p "${LOCAL_CACHE_ROOT}" "${LOCAL_EMOE_ROOT}" "${LOCAL_EXP_ROOT}"

# ── nuPlan env vars expected by devkit ────────────────────
# NOTE: with use_cache_without_dataset=true, raw dataset and maps are NOT needed

export NUPLAN_DATA_ROOT="${LOCAL_CACHE_ROOT}"
export NUPLAN_MAPS_ROOT="${LOCAL_CACHE_ROOT}/maps"
export NUPLAN_EXP_ROOT="${LOCAL_EXP_ROOT}"

echo "[INFO] NUPLAN_DATA_ROOT = ${NUPLAN_DATA_ROOT}"
echo "[INFO] NUPLAN_MAPS_ROOT = ${NUPLAN_MAPS_ROOT}"
echo "[INFO] NUPLAN_EXP_ROOT  = ${NUPLAN_EXP_ROOT}"

# ── Sync feature cache from S3 to local NVMe ──────────────
echo "[INFO] Syncing feature cache from S3..."
aws s3 sync \
    "${S3_DATA_ROOT}/exp/million" \
    "${LOCAL_CACHE_ROOT}/million" \
    --only-show-errors

echo "[INFO] Feature cache sync complete."

# ── Sync EMoE scene labels and anchors from S3 ────────────
echo "[INFO] Syncing EMoE scene labels and anchors from S3..."
aws s3 sync \
    "${S3_DATA_ROOT}/exp/emoe" \
    "${LOCAL_EMOE_ROOT}" \
    --only-show-errors

echo "[INFO] EMoE labels sync complete."
echo "[INFO] Contents of ${LOCAL_EMOE_ROOT}:"
ls -lh "${LOCAL_EMOE_ROOT}"

# ── Set GPU visibility ────────────────────────────────────
if [ "${NUM_GPUS}" -gt 0 ]; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    echo "[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
else
    export CUDA_VISIBLE_DEVICES=""
    echo "[INFO] Running on CPU only"
fi

# ── Run EMoE / PLUTO training ─────────────────────────────
echo "[INFO] Running EMoE training..."

TRAIN_CMD=(
    python /workspace/Thesis/run_training.py
    py_func=train
    +training=train_pluto
    worker=single_machine_thread_pool
    worker.max_workers="${MAX_WORKERS}"
    scenario_builder=nuplan
    cache.cache_path="${LOCAL_CACHE_ROOT}/million"
    cache.use_cache_without_dataset=true
    data_loader.params.batch_size="${BATCH_SIZE}"
    data_loader.params.num_workers="${NUM_WORKERS}"
    lr="${LR}"
    epochs="${EPOCHS}"
    warmup_epochs="${WARMUP_EPOCHS}"
    weight_decay="${WEIGHT_DECAY}"
)

# Append warm start checkpoint if provided
if [ -n "${WARM_START_CHECKPOINT_PATH}" ]; then
    TRAIN_CMD+=(checkpoint="${WARM_START_CHECKPOINT_PATH}")
fi

echo "[INFO] Training command:"
printf '  %q ' "${TRAIN_CMD[@]}"
echo

"${TRAIN_CMD[@]}"

echo "[INFO] Training finished successfully."

# ── Sync experiment outputs back to S3 ────────────────────
echo "[INFO] Syncing outputs back to S3..."
aws s3 sync \
    "${LOCAL_EXP_ROOT}" \
    "${S3_DATA_ROOT}/exp/million_training" \
    --only-show-errors

echo "[INFO] Done."
