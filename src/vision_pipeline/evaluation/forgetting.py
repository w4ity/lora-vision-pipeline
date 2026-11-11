"""Catastrophic-forgetting probe for the vision setting.

Fine-tuning on a domain (EuroSAT) can silently degrade the backbone's general
representations. We quantify this with two paired measurements taken from the
*original pretrained* backbone vs the *adapted* backbone, on a disjoint general
dataset (CIFAR-10, zero class overlap with EuroSAT):

1. **Feature drift** — mean cosine distance between paired penultimate features
   on the same CIFAR-10 inputs. 0.0 = representations untouched.
2. **kNN accuracy delta** — 10-NN classification on frozen features, base vs
   adapted. A large drop means general feature quality degraded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvd
from torchvision import transforms as T

from vision_pipeline.config import ModelConfig
from vision_pipeline.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from vision_pipeline.evaluation.evaluate import load_trained
from vision_pipeline.models.factory import build_classifier, resolve_device
from vision_pipeline.utils import get_logger, md_table

log = get_logger()


def _cifar_loaders(cifar_root: str | Path, img_size: int, n_train: int, n_test: int, seed: int):
    tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train = tvd.CIFAR10(root=str(cifar_root), train=True, download=True, transform=tf)
    test = tvd.CIFAR10(root=str(cifar_root), train=False, download=True, transform=tf)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train), size=min(n_train, len(train)), replace=False)
    te_idx = rng.choice(len(test), size=min(n_test, len(test)), replace=False)
    return (
        DataLoader(Subset(train, tr_idx.tolist()), batch_size=64, num_workers=2),
        DataLoader(Subset(test, te_idx.tolist()), batch_size=64, num_workers=2),
    )


@torch.inference_mode()
def extract_features(model, loader, device: torch.device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        f = model.features(x.to(device, non_blocking=True))
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def knn_accuracy(train_x, train_y, query_x, query_y, k: int = 10, batch: int = 512) -> float:
    def _norm(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    train_x, query_x = _norm(train_x.astype(np.float32)), _norm(query_x.astype(np.float32))
    correct = 0
    for start in range(0, len(query_x), batch):
        sims = query_x[start:start + batch] @ train_x.T
        topk = np.argpartition(-sims, k, axis=1)[:, :k]
        votes = train_y[topk]
        preds = np.array([np.bincount(row).argmax() for row in votes])
        correct += int((preds == query_y[start:start + batch]).sum())
    return correct / len(query_x)


def run_forgetting(
    run_dir: str | Path,
    n_train: int = 2000,
    n_test: int = 1000,
    cifar_root: str = "data/forgetting/cifar10",
    device: str = "auto",
    k: int = 10,
) -> Path:
    run_dir = Path(run_dir)
    dev = resolve_device(device)
    adapted, classes, cfg, meta = load_trained(run_dir, dev)

    # Base reference: the original pretrained backbone, untouched.
    base = build_classifier(
        ModelConfig(backbone=cfg.model.backbone, pretrained=True, method="linear_probe"),
        num_classes=len(classes),
    )
    base.to(dev).eval()

    train_loader, test_loader = _cifar_loaders(cifar_root, cfg.data.img_size, n_train, n_test, seed=cfg.seed)

    base_tr = extract_features(base, train_loader, dev)
    base_te = extract_features(base, test_loader, dev)
    ada_tr = extract_features(adapted, train_loader, dev)
    ada_te = extract_features(adapted, test_loader, dev)

    knn_base = knn_accuracy(*base_tr, *base_te, k=k)
    knn_adapted = knn_accuracy(*ada_tr, *ada_te, k=k)

    # Paired cosine distance on the same test inputs.
    def _norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    cos = (_norm(ada_te[0]) * _norm(base_te[0])).sum(axis=1)
    drift = float((1.0 - cos).mean())

    out_dir = run_dir / "forgetting"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ["kNN accuracy (CIFAR-10, k=%d)" % k, round(knn_base, 4), round(knn_adapted, 4),
         round(knn_adapted - knn_base, 4)],
        ["mean cosine feature distance (test)", "", round(drift, 4), ""],
    ]
    (out_dir / "results.json").write_text(json.dumps({
        "knn_base": knn_base, "knn_adapted": knn_adapted,
        "knn_delta": knn_adapted - knn_base, "mean_cosine_feature_distance": drift,
        "n_train": int(len(base_tr[1])), "n_test": int(len(base_te[1])),
    }, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        f"# Forgetting probe: {cfg.name}\n\n"
        f"Reference: original pretrained {cfg.model.backbone} vs adapted backbone, "
        f"measured on CIFAR-10 (disjoint from the training domain).\n\n"
        + md_table(["metric", "base", "adapted", "delta"], rows) + "\n\n"
        "Interpretation: a small kNN drop (< ~2 points) and low feature distance mean "
        "the fine-tune adapted the backbone without destroying general features.\n",
        encoding="utf-8",
    )
    log.info("Forgetting probe -> %s", out_dir / "report.md")
    print(md_table(["metric", "base", "adapted", "delta"], rows))
    return out_dir
