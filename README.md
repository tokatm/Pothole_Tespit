# Pothole D-FINE (Official Adaptation for Jetson Orin)

Bu klasor, resmi D-FINE reposu (`third_party/D-FINE`) uzerine ticari urun odakli pothole detection uyarlamasidir.

## Ne Hazirlandi?
- Resmi repo adaptasyonu: `third_party/D-FINE`
- Guvenlik kritik patchler:
  - `third_party/D-FINE/src/solver/det_engine.py`
  - `third_party/D-FINE/src/solver/det_solver.py`
  - `third_party/D-FINE/src/zoo/dfine/dfine_criterion.py`
- Orin + recall odakli config:
  - `third_party/D-FINE/configs/dfine/custom/pothole/dfine_hgnetv2_n_pothole_orin.yml`
- COCO dataset config:
  - `third_party/D-FINE/configs/dataset/pothole_detection_colab.yml`
- YOLO -> COCO donusum scripti:
  - `tools/yolo_to_coco_split.py`
- Colab scriptleri:
  - `scripts/colab_bootstrap.sh`
  - `scripts/train_colab.sh`
  - `scripts/export_onnx_orin.sh`

## Colab Setup Hucresi (`/content/pothole-dfine`)
```python
from google.colab import drive
drive.mount('/content/drive')

# Proje klasorunu Drive'dan /content/pothole-dfine altina kopyalayin
# Ornek: !cp -r /content/drive/MyDrive/pothole-dfine /content/pothole-dfine

%cd /content/pothole-dfine
!bash scripts/colab_bootstrap.sh
```

## YOLO -> COCO Donusumu
```bash
cd /content/pothole-dfine
python tools/yolo_to_coco_split.py \
  --labels-dir /content/dataset/labels \
  --images-dir /content/dataset/images \
  --output-dir /content/dataset/annotations \
  --val-ratio 0.2 \
  --class-names pothole
```

## Egitim
```bash
cd /content/pothole-dfine
bash scripts/train_colab.sh
```

## ONNX Export (Jetson Orin)
```bash
cd /content/pothole-dfine
bash scripts/export_onnx_orin.sh
```

## Not
- Bu repo TensorRT export hattina uygundur.
- INT8 icin Orin uzerinde calibration dataset ile ek calibration adimi gerekir.
