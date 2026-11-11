"""Hyperparameter sweep: train every config sequentially, evaluate each best
checkpoint on the shared test manifest, and emit a comparison table (CSV + MD).

The test split is only touched here, once per run, for final reporting — never
for selecting hyperparameters (that is what the val split is for).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

from vision_pipeline.config import load_config
from vision_pipeline.data.datasets import Manifest, ManifestDataset
from vision_pipeline.data.transforms import build_eval_transform
from vision_pipeline.evaluation.evaluate import evaluate_dataset, load_trained
from vision_pipeline.models.factory import resolve_device
from vision_pipeline.training.run import train_config
from vision_pipeline.utils import get_logger, md_table

log = get_logger()

SWEEP_FIELDS = [
    "name", "backbone", "method", "lora_rank", "lr", "epochs",
    "best_epoch", "val_macro_f1", "test_accuracy", "test_macro_f1", "train_minutes",
]


def run_sweep(config_paths: list[str | Path], out_dir: str | Path = "outputs/sweep") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    rows = []

    for path in config_paths:
        cfg = load_config(path)
        t0 = time.perf_counter()
        run_dir, best = train_config(cfg)
        minutes = (time.perf_counter() - t0) / 60

        # Final test evaluation with the best checkpoint.
        model, classes, cfg_r, meta = load_trained(run_dir, device)
        manifest = Manifest.load(
            Path(cfg_r.data.root) / "manifests" / cfg_r.data.name / "test.json"
        )
        ds = ManifestDataset(manifest, build_eval_transform(cfg_r.data.img_size))
        test_m = evaluate_dataset(model, ds, classes, device)

        rows.append({
            "name": cfg.name,
            "backbone": cfg.model.backbone,
            "method": cfg.model.method,
            "lora_rank": cfg.lora.rank if cfg.lora else "",
            "lr": cfg.train.lr,
            "epochs": cfg.train.epochs,
            "best_epoch": best["epoch"],
            "val_macro_f1": round(best["val_macro_f1"], 4),
            "test_accuracy": round(test_m["accuracy"], 4),
            "test_macro_f1": round(test_m["macro_f1"], 4),
            "train_minutes": round(minutes, 1),
        })

        (out_dir / "sweep_results.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    with open(out_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    table = md_table(
        SWEEP_FIELDS, [[r[k] for k in SWEEP_FIELDS] for r in rows]
    )
    (out_dir / "sweep_results.md").write_text(
        "# Sweep results\n\nValidation is used for model selection; test numbers are "
        "final-reporting only.\n\n" + table + "\n", encoding="utf-8"
    )
    log.info("Sweep complete -> %s", out_dir)
    print(table)
    return out_dir
