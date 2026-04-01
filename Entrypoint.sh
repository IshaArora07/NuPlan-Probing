#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting EMoE Training Container"

# ── PYTHONPATH ─────────────────────────────────────────────
export PYTHONPATH="/workspace/Thesis:/workspace/nuplan-devkit:${PYTHONPATH:-}"

# ── Required env vars ──────────────────────────────────────
# Example:
# s3://prod-pipeline/data/nuPlan_raw/dataset
: "${S3_DATA_ROOT:?ERROR: S3_DATA_ROOT is not set. Example: s3://prod-pipeline/data/nuPlan_raw/dataset}"

# ── Optional: local test mode ──────────────────────────────
LOCAL_TEST="${LOCAL_TEST:-false}"

# ── Training hyperparams ───────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MAX_WORKERS="${MAX_WORKERS:-32}"
CHECKPOINT="${CHECKPOINT:-}"
WANDB_MODE="${WANDB_MODE:-online}"

# ── Check AWS CLI ──────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
    echo "[ERROR] aws CLI not found in container"
    exit 1
fi

# ── Local paths ────────────────────────────────────────────
WORK_ROOT="/work"
LOCAL_CACHE_ROOT="${WORK_ROOT}/cache"
LOCAL_EMOE_ROOT="${WORK_ROOT}/emoe_labels"
LOCAL_EXP_ROOT="${WORK_ROOT}/experiments"

mkdir -p "${LOCAL_CACHE_ROOT}" "${LOCAL_EMOE_ROOT}" "${LOCAL_EXP_ROOT}"

# ── nuPlan env vars ────────────────────────────────────────
export NUPLAN_DATA_ROOT="${LOCAL_CACHE_ROOT}"
export NUPLAN_MAPS_ROOT="${LOCAL_CACHE_ROOT}/maps"
export NUPLAN_EXP_ROOT="${LOCAL_EXP_ROOT}"

echo "[INFO] NUPLAN_DATA_ROOT = ${NUPLAN_DATA_ROOT}"
echo "[INFO] NUPLAN_EXP_ROOT  = ${NUPLAN_EXP_ROOT}"

# ── Sync feature cache ─────────────────────────────────────
CACHE_S3_PATH="${S3_DATA_ROOT}/exp/million"
CACHE_LOCAL_PATH="${LOCAL_CACHE_ROOT}/million"

if [ "${LOCAL_TEST}" = "true" ]; then
    echo "[INFO] LOCAL_TEST=true — syncing small subset..."

    mkdir -p "${CACHE_LOCAL_PATH}"

    LOG_FOLDERS=$(aws s3 ls "${CACHE_S3_PATH}/" \
        | awk '{print $NF}' \
        | grep '/$' \
        | head -2 || true)

    if [ -z "${LOG_FOLDERS}" ]; then
        echo "[ERROR] No folders found in ${CACHE_S3_PATH}"
        exit 1
    fi

    for FOLDER in ${LOG_FOLDERS}; do
        echo "[INFO] Syncing ${FOLDER}"
        aws s3 sync "${CACHE_S3_PATH}/${FOLDER}" \
            "${CACHE_LOCAL_PATH}/${FOLDER}" \
            --only-show-errors
    done
else
    echo "[INFO] Syncing full feature cache..."
    aws s3 sync "${CACHE_S3_PATH}" \
        "${CACHE_LOCAL_PATH}" \
        --only-show-errors
fi

echo "[INFO] Cache sync complete."

# ── Sync EMoE labels + anchors ─────────────────────────────
EMOE_S3_PATH="${S3_DATA_ROOT}/exp/emoe"

echo "[INFO] Syncing EMoE labels..."
aws s3 sync "${EMOE_S3_PATH}" \
    "${LOCAL_EMOE_ROOT}" \
    --only-show-errors

ls -lh "${LOCAL_EMOE_ROOT}"

# ── EMoE env vars ──────────────────────────────────────────
export EMOE_SCENE_LABELS_PATH="${LOCAL_EMOE_ROOT}/scene_labels.jsonl"

# IMPORTANT: change if your actual filename is different
if [ -f "${LOCAL_EMOE_ROOT}/scene_anchors.npy" ]; then
    export EMOE_SCENE_ANCHORS_PATH="${LOCAL_EMOE_ROOT}/scene_anchors.npy"
else
    export EMOE_SCENE_ANCHORS_PATH="${LOCAL_EMOE_ROOT}/anchors.npy"
fi

echo "[INFO] EMOE_SCENE_LABELS_PATH  = ${EMOE_SCENE_LABELS_PATH}"
echo "[INFO] EMOE_SCENE_ANCHORS_PATH = ${EMOE_SCENE_ANCHORS_PATH}"

# ── Validate required files ────────────────────────────────
[ -f "${EMOE_SCENE_LABELS_PATH}" ] || {
    echo "[ERROR] Missing scene_labels.jsonl"
    exit 1
}

[ -f "${EMOE_SCENE_ANCHORS_PATH}" ] || {
    echo "[ERROR] Missing anchors file"
    exit 1
}

# ── GPU visibility ─────────────────────────────────────────
if [ "${NUM_GPUS}" -gt 0 ] 2>/dev/null; then
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    export CUDA_VISIBLE_DEVICES=""
    echo "[INFO] CPU mode"
fi

# ── Training command ───────────────────────────────────────
echo "[INFO] Launching training..."

TRAIN_CMD=(
    python /workspace/Thesis/run_training.py
    py_func=train
    +training=train_pluto
    worker=single_machine_thread_pool
    "worker.max_workers=${MAX_WORKERS}"
    scenario_builder=nuplan
    "cache.cache_path=${CACHE_LOCAL_PATH}"
    cache.use_cache_without_dataset=true
    "data_loader.params.batch_size=${BATCH_SIZE}"
    "data_loader.params.num_workers=${NUM_WORKERS}"
    "lr=${LR}"
    "epochs=${EPOCHS}"
    "warmup_epochs=${WARMUP_EPOCHS}"
    "weight_decay=${WEIGHT_DECAY}"
    "wandb.mode=${WANDB_MODE}"
)

# Resume checkpoint
# IMPORTANT: only keep this key if your Hydra config uses checkpoint=
if [ -n "${CHECKPOINT}" ]; then
    TRAIN_CMD+=("checkpoint=${CHECKPOINT}")
fi

echo "[INFO] Training command:"
printf '  %q' "${TRAIN_CMD[@]}"
echo

"${TRAIN_CMD[@]}"

echo "[INFO] Training completed."

# ── Sync outputs ───────────────────────────────────────────
if [ "${LOCAL_TEST}" = "true" ]; then
    echo "[INFO] LOCAL_TEST=true — skipping output sync."
    echo "[INFO] Outputs stored at ${LOCAL_EXP_ROOT}"
else
    echo "[INFO] Syncing outputs..."
    aws s3 sync "${LOCAL_EXP_ROOT}" \
        "${S3_DATA_ROOT}/exp/million_training" \
        --only-show-errors
fi

echo "[INFO] Done."
