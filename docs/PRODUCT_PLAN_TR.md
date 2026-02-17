# Pothole Detection - Ticari Urun Plani (Jetson Orin)

## 1) Urun Hedefi
- Guvenlik kritik pothole detection sistemi.
- Ana optimizasyon metrigi: Recall (cukur kacirmama).
- Edge deployment hedefi: Jetson Orin Nano/NX/AGX, TensorRT FP16/INT8.

## 2) Teknik Mimari
- Algoritma: D-FINE (resmi repo uyarlamasi).
- Backbone: HGNetv2 (D-FINE official).
- Egitim ortami: Google Colab (CUDA GPU).
- Cikti: `.pth` agirlik + `.onnx` + TensorRT engine.

## 3) Guvenlik Kritik Egitim Stratejisi
- `loss_vfl` agirligi yuksek tutulur.
- `class_weight` ile pothole sinifi pozitif ceza artirilir.
- Eval tarafinda precision'dan once recall/FN takibi zorunlu.
- Esik kalibrasyonu deployment oncesi recall hedefi bazli yapilir.

## 4) Veri Hatti
- Giris etiket formati: YOLO (`/content/dataset/labels`).
- Donusum: YOLO -> COCO (`instances_train.json`, `instances_val.json`).
- Sinif standardi:
  - `0 -> pothole` (YOLO)
  - COCO'da `category_id=1` olarak yazilir.

## 5) Colab Operasyon Akisi
1. Proje klasorunu `/content/pothole-dfine` altina yukle/kopyala.
2. `scripts/colab_bootstrap.sh` ile ortam ve path hazirla.
3. `tools/yolo_to_coco_split.py` ile dataset donusumunu calistir.
4. D-FINE train (`third_party/D-FINE/train.py`).
5. ONNX export (`third_party/D-FINE/tools/deployment/export_onnx.py`).

## 6) Deployment Akisi (Orin)
- ONNX -> TensorRT: `trtexec --fp16` veya INT8 calibration.
- Gercek yol testinde recall/FN benchmarki.
- Kamera pipeline latency hedefi: model ve input size bazli optimize edilir.

## 7) Ticari Urun Kontrol Noktalari
- Dataset governance: versiyonlama ve audit izi.
- Model governance: her release icin metrics + changelog.
- Safety gate: recall esigi altindaki modeller release edilmez.
- Runtime monitoring: sahada FN/FPS drift takibi.
