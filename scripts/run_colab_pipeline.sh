#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/content/Pothole_Tespit"

cd "${PROJECT_ROOT}"
bash scripts/colab_bootstrap.sh

python tools/yolo_to_coco_split.py \
  --labels-dir /content/dataset/labels \
  --images-dir /content/dataset/images \
  --output-dir /content/dataset/annotations \
  --val-ratio 0.2 \
  --class-names pothole

bash scripts/train_colab.sh
bash scripts/export_onnx_orin.sh

echo "[OK] Tum pipeline tamamlandi"
