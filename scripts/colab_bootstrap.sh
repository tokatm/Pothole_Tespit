#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/pothole-dfine"
DFINE_ROOT="${PROJECT_ROOT}/third_party/D-FINE"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] ${PROJECT_ROOT} bulunamadi."
  exit 1
fi

cd "${PROJECT_ROOT}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd "${DFINE_ROOT}"
python -m pip install -r requirements.txt || true

echo "[OK] Colab bootstrap tamamlandi"
