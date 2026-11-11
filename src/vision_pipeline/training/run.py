"""Single-run orchestration shared by scripts/train.py and scripts/sweep.py."""
from __future__ import annotations

from pathlib import Path

from vision_pipeline.config import ExperimentConfig
from vision_pipeline.data.datasets import build_loaders
from vision_pipeline.models.factory import (
    build_classifier,
    resolve_device,
    trainable_param_summary,
)
from vision_pipeline.models.lora import apply_lora
from vision_pipeline.training.tracker import make_tracker
from vision_pipeline.training.trainer import Trainer
from vision_pipeline.utils import get_logger, seed_everything

log = get_logger()


def train_config(cfg: ExperimentConfig) -> tuple[Path, dict]:
    """Train one experiment end-to-end. Returns (run_dir, best_info)."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_loader, val_loader, test_loader, classes = build_loaders(cfg.data, seed=cfg.seed)
    num_classes = len(classes)
    log.info(
        "Run %s | backbone=%s method=%s | %d classes | device=%s | config_hash=%s",
        cfg.name, cfg.model.backbone, cfg.model.method, num_classes, device,
        cfg.config_hash(),
    )

    model = build_classifier(cfg.model, num_classes=num_classes)
    if cfg.model.method == "lora":
        model = apply_lora(model, cfg.lora, cfg.model.backbone)
    model = model.to(device)
    summary = trainable_param_summary(model)
    log.info(
        "Trainable params: %(trainable_params)d / %(total_params)d (%(trainable_pct)s%%)",
        summary,
    )

    tracker = make_tracker(cfg.tracking, run_name=cfg.name)
    tracker.log_params(cfg.flat_params())
    tracker.log_params(summary)

    run_dir = Path(cfg.output_dir) / cfg.name
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        device=device,
        run_dir=run_dir,
        tracker=tracker,
    )
    best = trainer.fit()

    # Log the trainable-parameter summary as a final metric snapshot too.
    tracker.log_metrics(
        {"best_val_macro_f1": best["val_macro_f1"], "best_epoch": best["epoch"]},
        step=best["epoch"],
    )
    log.info("Done. Run dir: %s", run_dir)
    return run_dir, best
