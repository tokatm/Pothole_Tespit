from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoPotholeDataset(Dataset):
    def __init__(self, coco_json: str, images_root: str, image_size: Tuple[int, int] = (640, 640)) -> None:
        self.images_root = Path(images_root)
        self.image_size = image_size

        with open(coco_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images = data.get('images', [])
        self.annotations = data.get('annotations', [])

        self.ann_map: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            self.ann_map.setdefault(int(ann['image_id']), []).append(ann)

        self.tf = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.images_root / img_info['file_name']

        with Image.open(img_path).convert('RGB') as im:
            w, h = im.size
            image = self.tf(im)

        anns = self.ann_map.get(int(img_info['id']), [])
        labels: List[int] = []
        boxes: List[List[float]] = []

        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x1 = max(0.0, x / w)
            y1 = max(0.0, y / h)
            x2 = min(1.0, (x + bw) / w)
            y2 = min(1.0, (y + bh) / h)
            labels.append(int(ann['category_id']))
            boxes.append([x1, y1, x2, y2])

        if not labels:
            labels = [0]
            boxes = [[0.0, 0.0, 0.0, 0.0]]

        target = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
        }
        return image, [target]
