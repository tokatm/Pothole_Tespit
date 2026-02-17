from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.det_engine import evaluate, train_one_epoch


@dataclass
class SolverConfig:
    epochs: int
    lr: float
    weight_decay: float
    max_norm: float
    output_dir: str


class DetSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader,
        val_loader,
        cfg: SolverConfig,
        device: Optional[str] = None,
    ) -> None:
        # Single-GPU izolasyonu: distributed ve set_epoch bagimliligini kapat.
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.best_val_loss = float('inf')
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _maybe_set_epoch(self, loader: Any, epoch: int) -> None:
        sampler = getattr(loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

    def fit(self) -> None:
        for epoch in range(self.cfg.epochs):
            self._maybe_set_epoch(self.train_loader, epoch)

            train_stats = train_one_epoch(
                model=self.model,
                criterion=self.criterion,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                max_norm=self.cfg.max_norm,
            )
            val_stats = evaluate(
                model=self.model,
                criterion=self.criterion,
                dataloader=self.val_loader,
                device=self.device,
            )

            val_loss = float(val_stats.get('loss', 1e9))
            print(f'[Epoch {epoch:03d}] train={train_stats} val={val_stats}')

            last_path = Path(self.cfg.output_dir) / 'last.pth'
            torch.save({'model': self.model.state_dict(), 'epoch': epoch}, last_path)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = Path(self.cfg.output_dir) / 'best.pth'
                torch.save({'model': self.model.state_dict(), 'epoch': epoch}, best_path)
                print(f'Yeni en iyi model kaydedildi: {best_path}')


def load_solver_config(config_path: str) -> SolverConfig:
    with open(config_path, 'r', encoding='utf-8') as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    train = raw.get('train', {})
    return SolverConfig(
        epochs=int(train.get('epochs', 100)),
        lr=float(train.get('lr', 2e-4)),
        weight_decay=float(train.get('weight_decay', 1e-4)),
        max_norm=float(train.get('max_norm', 0.1)),
        output_dir=str(train.get('output_dir', '/content/pothole-dfine/outputs')),
    )
