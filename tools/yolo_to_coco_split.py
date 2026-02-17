from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("YOLO labels -> COCO json (train/val)")
    parser.add_argument("--labels-dir", type=str, required=True, help="/content/dataset/labels")
    parser.add_argument("--images-dir", type=str, required=True, help="/content/dataset/images")
    parser.add_argument("--output-dir", type=str, required=True, help="/content/dataset/annotations")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-names", nargs="*", default=["pothole"])
    return parser.parse_args()


def yolo_bbox_to_coco(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    bw = w * img_w
    bh = h * img_h
    x = xc * img_w - bw / 2.0
    y = yc * img_h - bh / 2.0
    return [x, y, bw, bh]


def resolve_image(images_dir: Path, stem: str) -> Path | None:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = sorted(images_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def convert_subset(label_files: List[Path], images_dir: Path, categories: List[Dict]) -> Dict:
    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for label_path in label_files:
        image_path = resolve_image(images_dir, label_path.stem)
        if image_path is None:
            continue

        with Image.open(image_path) as im:
            img_w, img_h = im.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": img_w,
                "height": img_h,
            }
        )

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls_idx = int(cls)
            bbox = yolo_bbox_to_coco(float(xc), float(yc), float(w), float(h), img_w, img_h)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_idx + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"Label dosyasi bulunamadi: {labels_dir}")

    random.seed(args.seed)
    random.shuffle(label_files)

    split_idx = int(len(label_files) * (1.0 - args.val_ratio))
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(args.class_names)]

    train_coco = convert_subset(train_files, images_dir, categories)
    val_coco = convert_subset(val_files, images_dir, categories)

    train_json = out_dir / "instances_train.json"
    val_json = out_dir / "instances_val.json"

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_coco, f)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_coco, f)

    print(f"[OK] Train: {train_json} | images={len(train_coco['images'])} annotations={len(train_coco['annotations'])}")
    print(f"[OK] Val:   {val_json} | images={len(val_coco['images'])} annotations={len(val_coco['annotations'])}")


if __name__ == "__main__":
    main()
