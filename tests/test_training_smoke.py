"""End-to-end smoke: train a tiny LoRA run on the synthetic dataset (CPU, seconds)."""
import json

import torch

from vision_pipeline.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoraConfig,
    ModelConfig,
    TrackingConfig,
    TrainConfig,
)
from vision_pipeline.data.datasets import ManifestDataset
from vision_pipeline.data.transforms import build_eval_transform, build_train_transform
from vision_pipeline.models.factory import build_classifier
from vision_pipeline.models.lora import apply_lora
from vision_pipeline.training.tracker import NullTracker
from vision_pipeline.training.trainer import Trainer
from torch.utils.data import DataLoader


def _tiny_loaders(manifests, batch=8):
    train = DataLoader(ManifestDataset(manifests["train"], build_train_transform(64)),
                       batch_size=batch, shuffle=True)
    val = DataLoader(ManifestDataset(manifests["val"], build_eval_transform(64)),
                     batch_size=batch, shuffle=False)
    return train, val


def test_smoke_lora_training(manifests, tmp_path):
    cfg = ExperimentConfig(
        name="smoke_lora",
        device="cpu",
        data=DataConfig(name="synthetic", img_size=64, batch_size=8, num_workers=0),
        model=ModelConfig(backbone="resnet18", pretrained=False, method="lora"),
        lora=LoraConfig(rank=2, alpha=4),
        train=TrainConfig(epochs=2, max_steps_per_epoch=2, lr=1e-3, head_lr=1e-3, amp=False),
        eval=EvalConfig(batch_size=8),
        tracking=TrackingConfig(backend="none"),
    )

    train_loader, val_loader = _tiny_loaders(manifests)
    model = build_classifier(cfg.model, num_classes=3)
    model = apply_lora(model, cfg.lora, cfg.model.backbone)

    trainer = Trainer(cfg, model, train_loader, val_loader, ["alpha", "beta", "gamma"],
                      torch.device("cpu"), tmp_path / "run", NullTracker())
    best = trainer.fit()

    run_dir = tmp_path / "run"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "adapter").exists()          # PEFT deployment artifact
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert 1 <= best["epoch"] <= 2

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["peft"] is True and len(meta["classes"]) == 3

    # reload the exact trained model through the same path evaluation uses
    from vision_pipeline.evaluation.evaluate import load_trained

    reloaded, classes, _cfg, _meta = load_trained(run_dir, torch.device("cpu"))
    assert classes == ["alpha", "beta", "gamma"]
    x, _y = next(iter(DataLoader(ManifestDataset(manifests["val"], build_eval_transform(64)),
                                 batch_size=2)))
    with torch.inference_mode():
        out = reloaded(x)
    assert out.shape == (2, 3)
