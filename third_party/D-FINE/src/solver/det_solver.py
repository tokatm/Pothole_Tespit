"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    @staticmethod
    def _maybe_set_epoch(dataloader, epoch: int):
        if hasattr(dataloader, "set_epoch") and callable(getattr(dataloader, "set_epoch")):
            dataloader.set_epoch(epoch)
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch") and callable(getattr(sampler, "set_epoch")):
            sampler.set_epoch(epoch)

    @staticmethod
    def _get_stop_epoch(dataloader):
        collate_fn = getattr(dataloader, "collate_fn", None)
        if collate_fn is not None and hasattr(collate_fn, "stop_epoch"):
            return collate_fn.stop_epoch
        return float("inf")

    @staticmethod
    def _to_scalar(x):
        if isinstance(x, (list, tuple)):
            return float(x[0]) if len(x) > 0 else 0.0
        return float(x)

    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        target_metric = "recall"

        if self.use_wandb:
            import wandb
            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        best_stat = {"epoch": -1, target_metric: -1e9}

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader,
                self.evaluator, self.device, self.last_epoch, self.use_wandb
            )
            best_stat["epoch"] = self.last_epoch
            best_stat[target_metric] = self._to_scalar(test_stats.get(target_metric, 0.0))
            print(f"best_stat ({target_metric}): {best_stat}")

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        stop_epoch = self._get_stop_epoch(self.train_dataloader)

        for epoch in range(start_epoch, args.epochs):
            self._maybe_set_epoch(self.train_dataloader, epoch)
            if dist_utils.is_dist_available_and_initialized():
                self._maybe_set_epoch(self.train_dataloader, epoch)

            if epoch == stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    collate_fn = getattr(self.train_dataloader, "collate_fn", None)
                    if collate_fn is not None and hasattr(collate_fn, "ema_restart_decay"):
                        self.ema.decay = collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer,
                self.device, epoch, epochs=args.epochs, max_norm=args.clip_max_norm,
                print_freq=args.print_freq, ema=self.ema, scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer,
                use_wandb=self.use_wandb, output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader,
                self.evaluator, self.device, epoch, self.use_wandb, output_dir=self.output_dir
            )

            # tensorboard logging
            for k, v in test_stats.items():
                if self.writer and dist_utils.is_main_process():
                    vals = v if isinstance(v, (list, tuple)) else [v]
                    for i, vi in enumerate(vals):
                        self.writer.add_scalar(f"Test/{k}_{i}", float(vi), epoch)

            # recall-only best checkpoint logic
            cur_recall = self._to_scalar(test_stats.get(target_metric, 0.0))
            if cur_recall > self._to_scalar(best_stat.get(target_metric, -1e9)):
                best_stat["epoch"] = epoch
                best_stat[target_metric] = cur_recall
                if self.output_dir:
                    if epoch >= stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

            print(f"best_stat ({target_metric}): {best_stat}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                import wandb
                wandb_logs = {}
                if "coco_eval_bbox" in test_stats:
                    for idx, metric_name in enumerate(metric_names):
                        vals = test_stats["coco_eval_bbox"] if isinstance(test_stats["coco_eval_bbox"], (list, tuple)) else [test_stats["coco_eval_bbox"]]
                        if idx < len(vals):
                            wandb_logs[f"metrics/{metric_name}"] = vals[idx]
                wandb_logs["metrics/recall"] = cur_recall
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))

    def val(self):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, self.val_dataloader,
            self.evaluator, self.device, epoch=-1, use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return
