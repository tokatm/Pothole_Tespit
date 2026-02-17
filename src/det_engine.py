from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch


class SmoothedValue:
    def __init__(self, window_size: int = 20) -> None:
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return float(d.median().item()) if len(d) > 0 else 0.0

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return float(d.mean().item()) if len(d) > 0 else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


class MetricLogger:
    def __init__(self, delimiter: str = '  ') -> None:
        self.meters: Dict[str, SmoothedValue] = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = float(v.item())
            self.meters[k].update(float(v))

    def __str__(self) -> str:
        parts = []
        for name, meter in self.meters.items():
            parts.append(f'{name}: {meter.avg:.4f} (global {meter.global_avg:.4f})')
        return self.delimiter.join(parts)


@dataclass
class Batch:
    images: torch.Tensor
    targets: Optional[Any]


def smart_unpack_batch(batch: Any, device: torch.device) -> Batch:
    """
    Smart Unpacking: dataloader ciktilarinda str/Tensor farkini tolere eder.
    Beklenen olasi formatlar:
    - (images, targets)
    - {'images': ..., 'targets': ...}
    - sadece images tensoru
    """
    images: Optional[torch.Tensor] = None
    targets: Optional[Any] = None

    if isinstance(batch, dict):
        images = batch.get('images') or batch.get('image')
        targets = batch.get('targets') or batch.get('target')
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2 and not isinstance(batch[0], str):
            images, targets = batch[0], batch[1]
        elif len(batch) >= 1:
            images = batch[0]
    elif isinstance(batch, torch.Tensor):
        images = batch

    if images is None:
        raise ValueError(f'Batch unpack edilemedi. Tip: {type(batch)}')
    if isinstance(images, str):
        raise TypeError('images alani str olamaz. Dataset __getitem__ kontrol edin.')

    images = images.to(device, non_blocking=True)

    if isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    elif isinstance(targets, list):
        casted: List[Any] = []
        for t in targets:
            if isinstance(t, dict):
                new_t = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        new_t[k] = v.to(device, non_blocking=True)
                    else:
                        new_t[k] = v
                casted.append(new_t)
            else:
                casted.append(t)
        targets = casted

    return Batch(images=images, targets=targets)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    print_freq: int = 20,
) -> Dict[str, float]:
    model.train()
    criterion.train()

    logger = MetricLogger()
    start = time.time()

    for step, raw_batch in enumerate(dataloader):
        batch = smart_unpack_batch(raw_batch, device=device)

        outputs = model(batch.images)
        loss_dict = criterion(outputs, batch.targets)
        losses = sum(loss for loss in loss_dict.values())

        if not math.isfinite(float(losses.item())):
            raise RuntimeError(f'Loss finite degil: {losses.item()}, details={loss_dict}')

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        log_items = {k: float(v.detach().item()) for k, v in loss_dict.items()}
        log_items['loss'] = float(losses.detach().item())
        logger.update(**log_items)

        if step % print_freq == 0:
            elapsed = time.time() - start
            print(f'[Epoch {epoch:03d} | Iter {step:05d}] {logger} | {elapsed:.1f}s')

    return {k: m.global_avg for k, m in logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: Iterable[Any],
    device: torch.device,
    print_freq: int = 50,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()

    logger = MetricLogger()

    for step, raw_batch in enumerate(dataloader):
        batch = smart_unpack_batch(raw_batch, device=device)
        outputs = model(batch.images)
        loss_dict = criterion(outputs, batch.targets)
        losses = sum(loss for loss in loss_dict.values())

        log_items = {k: float(v.detach().item()) for k, v in loss_dict.items()}
        log_items['loss'] = float(losses.detach().item())
        logger.update(**log_items)

        if step % print_freq == 0:
            print(f'[Eval Iter {step:05d}] {logger}')

    return {k: m.global_avg for k, m in logger.meters.items()}
