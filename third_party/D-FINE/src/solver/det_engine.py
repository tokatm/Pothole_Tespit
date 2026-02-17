"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes

try:
    from ..misc import MetricLogger, SmoothedValue
except Exception:
    import collections
    import datetime
    import time

    class SmoothedValue(object):
        """Track a series of values and provide access to smoothed values over a window."""

        def __init__(self, window_size=20, fmt=None):
            if fmt is None:
                fmt = "{median:.4f} ({global_avg:.4f})"
            self.deque = collections.deque(maxlen=window_size)
            self.total = 0.0
            self.count = 0
            self.fmt = fmt

        def update(self, value, n=1):
            self.deque.append(value)
            self.count += n
            self.total += value * n

        @property
        def median(self):
            d = torch.tensor(list(self.deque))
            return d.median().item()

        @property
        def avg(self):
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            return d.mean().item()

        @property
        def global_avg(self):
            return self.total / self.count

        @property
        def max(self):
            return max(self.deque)

        @property
        def value(self):
            return self.deque[-1]

        def __str__(self):
            return self.fmt.format(
                median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
            )

    class MetricLogger(object):
        def __init__(self, delimiter="\t"):
            self.meters = collections.defaultdict(SmoothedValue)
            self.delimiter = delimiter

        def add_meter(self, name, meter):
            self.meters[name] = meter

        def update(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if isinstance(v, (float, int)):
                    self.meters[k].update(v)

        def __str__(self):
            loss_str = []
            for name, meter in self.meters.items():
                loss_str.append("{}: {}".format(name, str(meter)))
            return self.delimiter.join(loss_str)

        def synchronize_between_processes(self):
            return

        def log_every(self, iterable, print_freq, header=None):
            i = 0
            if not header:
                header = ""
            start_time = time.time()
            end = time.time()
            iter_time = SmoothedValue(fmt="{avg:.4f}")
            data_time = SmoothedValue(fmt="{avg:.4f}")
            space_fmt = ":" + str(len(str(len(iterable)))) + "d"
            if torch.cuda.is_available():
                log_msg = self.delimiter.join(
                    [
                        header,
                        "[{0" + space_fmt + "}/{1}]",
                        "eta: {eta}",
                        "{meters}",
                        "time: {time}",
                        "data: {data}",
                        "max mem: {memory:.0f}",
                    ]
                )
            else:
                log_msg = self.delimiter.join(
                    [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
                )
            MB = 1024.0 * 1024.0
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)
                if i % print_freq == 0:
                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        print(
                            log_msg.format(
                                i,
                                len(iterable),
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB,
                            )
                        )
                    else:
                        print(
                            log_msg.format(
                                i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                            )
                        )
                i += 1
                end = time.time()
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


def _smart_unpack_batch(batch):
    """Handle dataloader outputs robustly (tuple/list/dict with optional string fields)."""
    if isinstance(batch, dict):
        samples = batch.get("samples", batch.get("images", batch.get("image", None)))
        targets = batch.get("targets", batch.get("target", None))
        return samples, targets

    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2 and not isinstance(batch[0], str):
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None

    if isinstance(batch, str):
        raise TypeError("Invalid batch type: got str. Dataset/loader should return tensor-like samples.")

    return batch, None


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = _smart_unpack_batch(batch)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        if isinstance(samples, str):
            raise TypeError("Dataloader returned string sample. Check dataset __getitem__ output.")
        samples = samples.to(device)
        if targets is None:
            targets = []
        if isinstance(targets, dict):
            targets = [targets]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples, targets = _smart_unpack_batch(batch)
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        if isinstance(samples, str):
            raise TypeError("Dataloader returned string sample. Check dataset __getitem__ output.")
        samples = samples.to(device)
        if targets is None:
            targets = []
        if isinstance(targets, dict):
            targets = [targets]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    stats["recall"] = [float(metrics.get("recall", 0.0))]
    stats["precision"] = [float(metrics.get("precision", 0.0))]
    stats["f1"] = [float(metrics.get("f1", 0.0))]
    return stats, coco_evaluator
