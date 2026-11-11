"""Evaluate a trained run: clean test split + per-corruption stress benchmark.

`load_trained` rebuilds the model from `meta.json` in the run directory and
loads `best.pt`. For LoRA runs this recreates the exact adapted structure and
loads the full adapted state dict; the `adapter/` directory (PEFT format) is
kept as the deployment artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vision_pipeline.config import config_from_dict
from vision_pipeline.data.datasets import Manifest, ManifestDataset
from vision_pipeline.data.transforms import build_eval_transform
from vision_pipeline.evaluation.corrupt import CORRUPTIONS, CorruptedDataset
from vision_pipeline.models.factory import resolve_device
from vision_pipeline.models.lora import LoraConfig, apply_lora
from vision_pipeline.training.metrics import summarize
from vision_pipeline.utils import get_logger, md_table

log = get_logger()


def load_trained(run_dir: str | Path, device: torch.device | None = None):
    """Rebuild a trained model from its run directory. Returns (model, classes, cfg, meta)."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — is this a finished run directory?")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = config_from_dict(meta["config"])
    classes = meta["classes"]

    model = build_model_for_run(cfg, num_classes=len(classes), peft=meta.get("peft", False))
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    device = device or resolve_device("auto")
    model.to(device)
    model.eval()
    return model, classes, cfg, meta


def build_model_for_run(cfg, num_classes: int, peft: bool):
    from vision_pipeline.models.factory import build_classifier

    model = build_classifier(cfg.model, num_classes=num_classes)
    if peft:
        lora_cfg = LoraConfig(**(cfg.lora.__dict__ if cfg.lora else {}))
        model = apply_lora(model, lora_cfg, cfg.model.backbone)
    return model


@torch.inference_mode()
def evaluate_dataset(model, dataset, classes: list[str], device: torch.device,
                     batch_size: int = 64, num_workers: int = 2) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    logits_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(y.numpy())
    return summarize(np.concatenate(logits_all), np.concatenate(labels_all), classes)


def plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_run(
    run_dir: str | Path,
    split: str = "test",
    severities: list[int] | None = None,
    device: str = "auto",
    num_workers: int = 2,
) -> dict:
    run_dir = Path(run_dir)
    severities = severities or [1, 3, 5]
    dev = resolve_device(device)
    model, classes, cfg, meta = load_trained(run_dir, dev)

    manifest_path = Path(cfg.data.root) / "manifests" / cfg.data.name / f"{split}.json"
    manifest = Manifest.load(manifest_path)
    eval_tf = build_eval_transform(cfg.data.img_size)

    # ---- clean evaluation ---- #
    clean = evaluate_dataset(
        model, ManifestDataset(manifest, eval_tf), classes, dev, num_workers=num_workers
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{split}_metrics.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8"
    )
    plot_confusion(
        np.array(clean["confusion_matrix"]), classes, eval_dir / f"confusion_{split}.png"
    )
    log.info(
        "[%s] %s clean: acc %.4f | macro_f1 %.4f",
        cfg.name, split, clean["accuracy"], clean["macro_f1"],
    )

    # ---- stress benchmark (corruption x severity) ---- #
    raw_ds = ManifestDataset(manifest, transform=None)  # PIL access for corruption
    rows = []
    for corruption in CORRUPTIONS:
        for sev in severities:
            ds = CorruptedDataset(raw_ds, corruption, sev, eval_tf)
            m = evaluate_dataset(model, ds, classes, dev, num_workers=num_workers)
            rows.append({
                "corruption": corruption, "severity": sev,
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
            })
    (eval_dir / "stress_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    mean_stress_acc = float(np.mean([r["accuracy"] for r in rows]))
    table = md_table(
        ["corruption", "severity", "accuracy", "macro_f1"],
        [[r["corruption"], r["severity"], r["accuracy"], r["macro_f1"]] for r in rows],
    )
    (eval_dir / "stress.md").write_text(
        f"# Stress benchmark ({split} split)\n\n"
        f"Clean accuracy: **{clean['accuracy']:.4f}** | "
        f"Mean stress accuracy: **{mean_stress_acc:.4f}**\n\n{table}\n",
        encoding="utf-8",
    )
    log.info("[%s] mean stress accuracy: %.4f", cfg.name, mean_stress_acc)
    print(table)
    return clean
