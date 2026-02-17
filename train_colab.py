from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import CocoPotholeDataset
from src.det_solver import DetSolver, load_solver_config
from src.modeling import DFineLikeDetector, DetectionCriterion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('Train pothole detector in Colab')
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1][0] for b in batch]
    return images, targets


def main() -> None:
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    image_size = tuple(cfg['model'].get('input_size', [640, 640]))
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('train', {})
    loss_cfg = cfg.get('loss', {})

    train_ds = CocoPotholeDataset(
        coco_json=data_cfg['train_json'],
        images_root=data_cfg['images_root'],
        image_size=(image_size[0], image_size[1]),
    )

    val_ds = CocoPotholeDataset(
        coco_json=data_cfg.get('val_json', data_cfg['train_json']),
        images_root=data_cfg['images_root'],
        image_size=(image_size[0], image_size[1]),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get('batch_size', 16)),
        shuffle=True,
        num_workers=int(train_cfg.get('num_workers', 2)),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get('batch_size', 16)),
        shuffle=False,
        num_workers=int(train_cfg.get('num_workers', 2)),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = DFineLikeDetector(num_classes=int(cfg['model'].get('num_classes', 2)))
    criterion = DetectionCriterion(
        lambda_cls=float(loss_cfg.get('lambda_cls', 3.5)),
        lambda_bbox=float(loss_cfg.get('lambda_bbox', 1.0)),
        lambda_giou=float(loss_cfg.get('lambda_giou', 0.0)),
        vfocal_alpha=float(loss_cfg.get('vfocal_alpha', 0.9)),
        vfocal_gamma=float(loss_cfg.get('vfocal_gamma', 2.0)),
        class_weight=loss_cfg.get('class_weight', [1.0, 4.5]),
    )

    solver_cfg = load_solver_config(args.config)
    solver = DetSolver(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=solver_cfg,
        device=cfg.get('runtime', {}).get('device', 'cuda'),
    )
    solver.fit()


if __name__ == '__main__':
    main()
