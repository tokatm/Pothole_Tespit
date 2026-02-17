from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHGNetV2Backbone(nn.Module):
    """
    Yer tutucu hafif backbone.
    Gercek projede HGNetv2 implementasyonu ile degistirilmelidir.
    TensorRT uyumlu katmanlar kullanilir (Conv/BN/ReLU/AdaptivePool).
    """

    def __init__(self, in_ch: int = 3, width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        return x.flatten(1)


class DFineLikeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = SimpleHGNetV2Backbone()
        self.neck = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(images)
        feat = self.neck(feat)
        logits = self.cls_head(feat)
        boxes = self.box_head(feat).sigmoid()
        return {'pred_logits': logits, 'pred_boxes': boxes}


class VarifocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        class_weight: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weight is None:
            class_weight = [1.0, 3.0]
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = F.one_hot(labels, num_classes=logits.shape[-1]).float()

        focal_weight = self.alpha * (probs - targets).abs().pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        cw = self.class_weight.view(1, -1)
        loss = focal_weight * bce * cw
        return loss.mean()


class DetectionCriterion(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 3.0,
        lambda_bbox: float = 1.0,
        lambda_giou: float = 0.0,
        vfocal_alpha: float = 0.9,
        vfocal_gamma: float = 2.0,
        class_weight: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.vfl = VarifocalLoss(alpha=vfocal_alpha, gamma=vfocal_gamma, class_weight=class_weight)

    def _extract_labels_boxes(self, targets: Any, device: torch.device, bs: int) -> Any:
        labels = torch.zeros(bs, dtype=torch.long, device=device)
        boxes = torch.zeros(bs, 4, dtype=torch.float32, device=device)

        if isinstance(targets, torch.Tensor):
            return targets[:, 0].long(), targets[:, 1:5].float()

        if isinstance(targets, list):
            for i, t in enumerate(targets[:bs]):
                if isinstance(t, dict):
                    l = t.get('labels')
                    b = t.get('boxes')
                    if isinstance(l, torch.Tensor) and l.numel() > 0:
                        labels[i] = l[0].long()
                    if isinstance(b, torch.Tensor) and b.numel() >= 4:
                        boxes[i] = b[0].float()
        return labels, boxes

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Any) -> Dict[str, torch.Tensor]:
        logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        bs = logits.shape[0]
        labels, gt_boxes = self._extract_labels_boxes(targets, logits.device, bs)

        loss_cls = self.vfl(logits, labels)
        loss_bbox = F.l1_loss(pred_boxes, gt_boxes)

        total_cls = self.lambda_cls * loss_cls
        total_bbox = self.lambda_bbox * loss_bbox
        total_giou = torch.tensor(0.0, device=logits.device) * self.lambda_giou

        return {
            'loss_cls': total_cls,
            'loss_bbox': total_bbox,
            'loss_giou': total_giou,
        }
