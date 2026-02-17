#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/pothole-dfine"
DFINE_ROOT="${PROJECT_ROOT}/third_party/D-FINE"
CONFIG="configs/dfine/custom/pothole/dfine_hgnetv2_n_pothole_orin.yml"
CHECKPOINT="/content/pothole-dfine/outputs/dfine_hgnetv2_n_pothole_orin/best_stg1.pth"

cd "${DFINE_ROOT}"
python tools/deployment/export_onnx.py --check -c "${CONFIG}" -r "${CHECKPOINT}"

echo "[OK] ONNX export tamamlandi"
