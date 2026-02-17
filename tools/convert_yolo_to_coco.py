from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('YOLO to COCO converter')
    p.add_argument('--labels-dir', type=str, required=True)
    p.add_argument('--images-dir', type=str, required=True)
    p.add_argument('--output-json', type=str, required=True)
    p.add_argument('--class-names', nargs='*', default=['background', 'pothole'])
    p.add_argument('--default-ext', type=str, default='jpg')
    return p.parse_args()


def yolo_to_coco_bbox(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    x = (xc * img_w) - (bw / 2.0)
    y = (yc * img_h) - (bh / 2.0)
    return x, y, bw, bh


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [{'id': i, 'name': name} for i, name in enumerate(args.class_names)]

    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    image_id = 1

    for label_file in sorted(labels_dir.glob('*.txt')):
        stem = label_file.stem
        image_path = images_dir / f'{stem}.{args.default_ext}'
        if not image_path.exists():
            candidates = list(images_dir.glob(f'{stem}.*'))
            if not candidates:
                continue
            image_path = candidates[0]

        with Image.open(image_path) as im:
            img_w, img_h = im.size

        images.append(
            {
                'id': image_id,
                'file_name': image_path.name,
                'width': img_w,
                'height': img_h,
            }
        )

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # YOLO siniflari genelde 0'dan baslar. 0'i background icin ayirdigimizdan +1 offset uygula.
                cls = float(parts[0]) + 1
                xc, yc, w, h = map(float, parts[1:])
                x, y, bw, bh = yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h)

                annotations.append(
                    {
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': cls,
                        'bbox': [x, y, bw, bh],
                        'area': bw * bh,
                        'iscrowd': 0,
                    }
                )
                ann_id += 1

        image_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f'COCO json yazildi: {out_path}')
    print(f'images={len(images)}, annotations={len(annotations)}')


if __name__ == '__main__':
    main()
