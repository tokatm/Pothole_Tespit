#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/Pothole_Tespit"
DFINE_ROOT="${PROJECT_ROOT}/third_party/D-FINE"
CONFIG="configs/dfine/custom/pothole/dfine_hgnetv2_n_pothole_orin.yml"

cd "${PROJECT_ROOT}"
python tools/colab_setup.py

cd "${DFINE_ROOT}"
python train.py -c "${CONFIG}" --seed=42

echo "[OK] Egitim tamamlandi"
