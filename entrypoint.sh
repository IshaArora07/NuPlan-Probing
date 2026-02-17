#!/usr/bin/env bash
set -euo pipefail

# Make emoe importable even if it isn't an installable package
export PYTHONPATH="/workspace/emoe:/workspace/nuplan-devkit:${PYTHONPATH:-}"

# Expected env vars (fail fast if missing)
: "${S3_INPUT:?Set S3_INPUT like s3://prod-pipeline/data/nuplan/...}"
: "${S3_OUTPUT:?Set S3_OUTPUT like s3://prod-pipeline/data/nuplan_outputs/...}"

# Local scratch inside container
WORKDIR="/work"
INPUT_DIR="${WORKDIR}/input"
OUTPUT_DIR="${WORKDIR}/output"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

echo "[INFO] S3_INPUT=${S3_INPUT}"
echo "[INFO] S3_OUTPUT=${S3_OUTPUT}"

# Pull only what you need locally before classification.
# If your classifier can stream from S3, you can remove this section.
echo "[INFO] Syncing input from S3 to local..."
aws s3 sync "${S3_INPUT}" "${INPUT_DIR}" --only-show-errors

echo "[INFO] Running classification..."
# Replace this with your actual classification entrypoint.
# Examples:
# python /workspace/emoe/scripts/run_classification.py --data_root "${INPUT_DIR}" --out_dir "${OUTPUT_DIR}"
# python /workspace/emoe/tools/classify_nuplan.py --input "${INPUT_DIR}" --output "${OUTPUT_DIR}"

python /workspace/emoe/run_classification.py \
  --data_root "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "[INFO] Syncing outputs back to S3..."
aws s3 sync "${OUTPUT_DIR}" "${S3_OUTPUT}" --only-show-errors

echo "[INFO] Done."
