"""Training loop: warmup+cosine schedule, dual LR (adapters vs head), AMP,
early stopping on val macro-F1, checkpointing (best.pt + PEFT adapter dir),
per-epoch CSV + JSON history, MLflow logging.

Reproducibility contract: everything needed to rebuild the best model lives in
the run directory — best.pt, adapter/ (if LoRA), meta.json (resolved config),
metrics.csv, history.json.
"""
from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vision_pipeline.config import ExperimentConfig
from vision_pipeline.models.lora import configure_train_mode, is_peft_model
from vision_pipeline.training.metrics import summarize
from vision_pipeline.training.tracker import NullTracker
from vision_pipeline.utils import get_logger

log = get_logger()

CSV_FIELDS = [
    "epoch", "lr", "train_loss", "train_acc",
    "val_loss", "val_acc", "val_macro_f1", "val_top5_acc", "epoch_s", "gpu_mem_gb",
]


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        classes: list[str],
        device: torch.device,
        run_dir: str | Path,
        tracker=None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = classes
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker or NullTracker()
        self.is_peft = is_peft_model(model)

        # Param groups: LoRA adapters at cfg.train.lr, everything else trainable
        # (the fresh head, linear-probe heads) at cfg.train.head_lr.
        lora_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if "lora_" in name else head_params).append(p)
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": cfg.train.lr, "name": "lora"})
        if head_params:
            groups.append({"params": head_params, "lr": cfg.train.head_lr, "name": "head"})
        if not groups:
            raise RuntimeError("No trainable parameters — check method/target config.")
        self.optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)

        steps_per_epoch = len(train_loader)
        if cfg.train.max_steps_per_epoch:
            steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
        total_steps = max(cfg.train.epochs * steps_per_epoch, 1)
        warmup_steps = max(int(cfg.train.warmup_epochs * steps_per_epoch), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        self.amp_enabled = cfg.train.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.history: list[dict] = []
        self.csv_path = self.run_dir / "metrics.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_FIELDS)

        self.best = {"epoch": -1, "val_macro_f1": -1.0, "val_accuracy": -1.0}

    # ------------------------------------------------------------------ #
    def fit(self) -> dict:
        cfg = self.cfg.train
        no_improve = 0
        log.info(
            "Training %s | %d epochs | device=%s | peft=%s",
            self.cfg.name, cfg.epochs, self.device, self.is_peft,
        )
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            train_m = self._train_epoch(epoch)
            val_m = self._validate()
            epoch_s = time.perf_counter() - t0

            lr_now = self.optimizer.param_groups[0]["lr"]
            gpu_mem = (
                torch.cuda.max_memory_allocated() / 1e9
                if self.device.type == "cuda"
                else 0.0
            )
            row = {
                "epoch": epoch, "lr": lr_now,
                "train_loss": train_m["loss"], "train_acc": train_m["acc"],
                "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"], "val_top5_acc": val_m["top5_accuracy"],
                "epoch_s": round(epoch_s, 2), "gpu_mem_gb": round(gpu_mem, 2),
            }
            self.history.append(row)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row[k] for k in CSV_FIELDS])
            self.tracker.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"}, step=epoch
            )
            log.info(
                "epoch %d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f f1 %.4f | %.1fs",
                epoch, row["train_loss"], row["train_acc"], row["val_loss"],
                row["val_acc"], row["val_macro_f1"], epoch_s,
            )
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            if val_m["macro_f1"] > self.best["val_macro_f1"] + 1e-4:
                self.best = {
                    "epoch": epoch,
                    "val_macro_f1": val_m["macro_f1"],
                    "val_accuracy": val_m["accuracy"],
                }
                self._save_best()
                no_improve = 0
                log.info("  new best val_macro_f1 %.4f -> saved", val_m["macro_f1"])
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    log.info("Early stopping at epoch %d (no improvement in %d epochs).",
                             epoch, no_improve)
                    break

        self._write_meta()
        self.tracker.log_artifact(self.csv_path)
        self.tracker.finish()
        (self.run_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        log.info("Best: epoch %d, val_macro_f1 %.4f", self.best["epoch"], self.best["val_macro_f1"])
        return self.best

    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch: int) -> dict:
        configure_train_mode(self.model, self.cfg.model.method)
        steps = len(self.train_loader)
        if self.cfg.train.max_steps_per_epoch:
            steps = min(steps, self.cfg.train.max_steps_per_epoch)
        total, correct, loss_sum = 0, 0, 0.0

        pbar = tqdm(self.train_loader, total=steps, desc=f"epoch {epoch}", leave=False)
        for step, (x, y) in enumerate(pbar):
            if step >= steps:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.cfg.train.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            pbar.set_postfix(loss=f"{loss_sum / max(total, 1):.4f}",
                             acc=f"{correct / max(total, 1):.4f}")
        return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}

    @torch.inference_mode()
    def _validate(self) -> dict:
        self.model.eval()
        logits_all, labels_all, loss_sum, total = [], [], 0.0, 0
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss_sum += self.criterion(logits, y).item() * len(y)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(y.cpu().numpy())
            total += len(y)
        logits_np = np.concatenate(logits_all)
        labels_np = np.concatenate(labels_all)
        return summarize(logits_np, labels_np, self.classes, loss=loss_sum / max(total, 1))

    # ------------------------------------------------------------------ #
    def _save_best(self) -> None:
        if self.is_peft:
            self.model.save_pretrained(self.run_dir / "adapter")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": self.best["epoch"],
                "val_macro_f1": self.best["val_macro_f1"],
            },
            self.run_dir / "best.pt",
        )

    def _write_meta(self) -> None:
        meta = {
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash(),
            "classes": self.classes,
            "peft": self.is_peft,
            "best": self.best,
            "torch_version": torch.__version__,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
