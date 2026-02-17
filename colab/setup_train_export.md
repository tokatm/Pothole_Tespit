```python
from google.colab import drive
drive.mount('/content/drive')

# 1) Proje klasorunu /content/pothole-dfine altina alin
# !cp -r /content/drive/MyDrive/pothole-dfine /content/pothole-dfine

%cd /content/pothole-dfine
!bash scripts/colab_bootstrap.sh
```

```bash
# 2) YOLO -> COCO
cd /content/pothole-dfine
python tools/yolo_to_coco_split.py \
  --labels-dir /content/dataset/labels \
  --images-dir /content/dataset/images \
  --output-dir /content/dataset/annotations \
  --val-ratio 0.2 \
  --class-names pothole
```

```bash
# 3) Train
cd /content/pothole-dfine
bash scripts/train_colab.sh
```

```bash
# 4) ONNX Export
cd /content/pothole-dfine
bash scripts/export_onnx_orin.sh
```
