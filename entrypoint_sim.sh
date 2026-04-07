Those triple backticks make the script **invalid bash immediately**.

Below is the **fully corrected final version**, including your new **RENDER override support**.

---

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Starting EMoE Simulation Container"

# ── PYTHONPATH ─────────────────────────────────────────────
export PYTHONPATH="/workspace/Thesis:/workspace/nuplan-devkit:${PYTHONPATH:-}"

# ── Required env vars ──────────────────────────────────────
: "${S3_DATA_ROOT:?ERROR: S3_DATA_ROOT is not set. Example: s3://prod-pipeline/data/nuPlan_raw/dataset}"
: "${S3_CHECKPOINT_PATH:?ERROR: S3_CHECKPOINT_PATH is not set. Example: s3://prod-pipeline/data/nuPlan_raw/exp/checkpoints/emoe.ckpt}"

# ── Optional: local test mode ──────────────────────────────
LOCAL_TEST="${LOCAL_TEST:-false}"

# ── Simulation params ──────────────────────────────────────
SCENARIO_BUILDER="${SCENARIO_BUILDER:-nuplan}"
SCENARIO_FILTER="${SCENARIO_FILTER:-val14_benchmark}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-emoe_simulation}"
RENDER="${RENDER:-false}"
VIDEO_SAVE_DIR="${VIDEO_SAVE_DIR:-/tmp/dummy}"

# ── Local test params ──────────────────────────────────────
LOCAL_TEST_SCENARIO_BUILDER="${LOCAL_TEST_SCENARIO_BUILDER:-nuplan_mini}"
LOCAL_TEST_SCENARIO_FILTER="${LOCAL_TEST_SCENARIO_FILTER:-mini_demo_scenario}"
LOCAL_TEST_S3_DB_PATH="${LOCAL_TEST_S3_DB_PATH:-${S3_DATA_ROOT}/nuplan-v1.1/mini}"

# ── Full run DB path ───────────────────────────────────────
FULL_S3_DB_PATH="${FULL_S3_DB_PATH:-${S3_DATA_ROOT}/nuplan-v1.1/trainval}"

# ── Check AWS CLI ──────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
    echo "[ERROR] aws CLI not found in container"
    exit 1
fi

# ── Local paths ────────────────────────────────────────────
WORK_ROOT="/work"
LOCAL_CACHE_ROOT="${WORK_ROOT}/cache"
LOCAL_EXP_ROOT="${WORK_ROOT}/experiments"
CKPT_DIR="/workspace/Thesis/checkpoints"

mkdir -p "${LOCAL_CACHE_ROOT}" "${LOCAL_EXP_ROOT}" "${CKPT_DIR}"

# ── nuPlan env vars ────────────────────────────────────────
export NUPLAN_DATA_ROOT="${LOCAL_CACHE_ROOT}"
export NUPLAN_MAPS_ROOT="${LOCAL_CACHE_ROOT}/maps"
export NUPLAN_EXP_ROOT="${LOCAL_EXP_ROOT}"

echo "[INFO] NUPLAN_DATA_ROOT = ${NUPLAN_DATA_ROOT}"
echo "[INFO] NUPLAN_EXP_ROOT  = ${NUPLAN_EXP_ROOT}"

# ── Sync DB files ──────────────────────────────────────────
if [ "${LOCAL_TEST}" = "true" ]; then
    echo "[INFO] LOCAL_TEST=true — syncing mini dataset..."

    DB_LOCAL_PATH="${LOCAL_CACHE_ROOT}/nuplan-v1.1/mini"
    mkdir -p "${DB_LOCAL_PATH}"

    aws s3 sync \
        "${LOCAL_TEST_S3_DB_PATH}" \
        "${DB_LOCAL_PATH}" \
        --only-show-errors

    ACTIVE_SCENARIO_BUILDER="${LOCAL_TEST_SCENARIO_BUILDER}"
    ACTIVE_SCENARIO_FILTER="${LOCAL_TEST_SCENARIO_FILTER}"
else
    echo "[INFO] Syncing full val DB split..."

    DB_LOCAL_PATH="${LOCAL_CACHE_ROOT}/nuplan-v1.1/trainval"
    mkdir -p "${DB_LOCAL_PATH}"

    aws s3 sync \
        "${FULL_S3_DB_PATH}" \
        "${DB_LOCAL_PATH}" \
        --only-show-errors

    ACTIVE_SCENARIO_BUILDER="${SCENARIO_BUILDER}"
    ACTIVE_SCENARIO_FILTER="${SCENARIO_FILTER}"
fi

echo "[INFO] DB sync complete."

# ── Sync maps ──────────────────────────────────────────────
MAPS_S3_PATH="${S3_DATA_ROOT}/maps"

echo "[INFO] Syncing nuPlan maps..."
aws s3 sync \
    "${MAPS_S3_PATH}" \
    "${LOCAL_CACHE_ROOT}/maps" \
    --only-show-errors

echo "[INFO] Maps sync complete."

export NUPLAN_DB_FILES="${DB_LOCAL_PATH}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

echo "[INFO] NUPLAN_DB_FILES    = ${NUPLAN_DB_FILES}"
echo "[INFO] NUPLAN_MAP_VERSION = ${NUPLAN_MAP_VERSION}"
echo "[INFO] RENDER             = ${RENDER}"

# ── Download checkpoint from S3 ────────────────────────────
CHECKPOINT_FILENAME="$(basename "${S3_CHECKPOINT_PATH}")"
LOCAL_CHECKPOINT="${CKPT_DIR}/${CHECKPOINT_FILENAME}"

echo "[INFO] Downloading checkpoint..."
aws s3 cp \
    "${S3_CHECKPOINT_PATH}" \
    "${LOCAL_CHECKPOINT}" \
    --only-show-errors

if [ ! -f "${LOCAL_CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found at ${LOCAL_CHECKPOINT}"
    exit 1
fi

echo "[INFO] Checkpoint ready: ${LOCAL_CHECKPOINT}"

# ── Simulation command ─────────────────────────────────────
echo "[INFO] Launching simulation..."

cd /workspace/Thesis

sh ./script/run_pluto_planner.sh \
    pluto_planner \
    "${ACTIVE_SCENARIO_BUILDER}" \
    "${ACTIVE_SCENARIO_FILTER}" \
    "${CHECKPOINT_FILENAME}" \
    "${VIDEO_SAVE_DIR}" \
    "planner.pluto_planner.render=${RENDER}"

echo "[INFO] Simulation completed."

# ── Sync outputs ───────────────────────────────────────────
if [ "${LOCAL_TEST}" = "true" ]; then
    echo "[INFO] LOCAL_TEST=true — skipping output sync."
    echo "[INFO] Outputs stored at ${LOCAL_EXP_ROOT}"
else
    OUTPUT_S3_PATH="${S3_DATA_ROOT}/exp/simulation_results/${EXPERIMENT_NAME}"

    echo "[INFO] Syncing outputs..."
    aws s3 sync \
        "${LOCAL_EXP_ROOT}" \
        "${OUTPUT_S3_PATH}" \
        --only-show-errors
fi

echo "[INFO] Done."
