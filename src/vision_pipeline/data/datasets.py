"""Dataset materialization, stratified splitting and manifest-driven loading.

Design notes
------------
* Every supported dataset is *materialized* to plain image files on disk under
  ``data/raw/<name>/`` so the pipeline treats all datasets uniformly.
* Splits are written as JSON **manifests** (``data/manifests/<name>/{train,val,test}.json``)
  with a fixed seed. The manifest is frozen before training starts: nothing can
  leak between splits, and any run can be traced back to the exact data it saw.
* Known caveat (documented, not hidden): EuroSAT patches come from overlapping
  Sentinel-2 scenes, so a purely random split can share scenes across splits.
  We keep the standard random-stratified split for comparability and flag this
  as a limitation — a scene-aware split is a natural extension.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tvd

from vision_pipeline.config import DataConfig
from vision_pipeline.data.transforms import build_eval_transform, build_train_transform

log = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# --------------------------------------------------------------------------- #
# Materialization: download once, expose a flat list of (path, label) items.
# --------------------------------------------------------------------------- #
def materialize(name: str, root: str | Path, download: bool = True):
    """Return (items, classes) where items = [(path, class_name), ...]."""
    root = Path(root)
    raw = root / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)

    if name == "eurosat":
        ds = tvd.EuroSAT(root=str(raw), download=download)
        classes = list(ds.classes)
        items = [(str(p), classes[i]) for p, i in ds.samples]

    elif name == "cifar10":
        classes = list(CIFAR10_CLASSES)
        items = []
        for train in (True, False):
            ds = tvd.CIFAR10(root=str(raw), train=train, download=download)
            tag = "train" if train else "test"
            for idx, (arr, y) in enumerate(zip(ds.data, ds.targets)):
                cls_dir = raw / "images" / classes[y]
                cls_dir.mkdir(parents=True, exist_ok=True)
                out = cls_dir / f"cifar_{tag}_{idx:05d}.png"
                if not out.exists():  # one-time PNG dump, idempotent
                    Image.fromarray(arr).save(out)
                items.append((str(out), classes[y]))

    elif name == "flowers102":
        classes = [f"class_{i:03d}" for i in range(102)]
        items = []
        for split in ("train", "val", "test"):
            ds = tvd.Flowers102(root=str(raw), split=split, download=download)
            for idx in range(len(ds)):
                img, y = ds[idx]
                cls_dir = raw / "images" / classes[y]
                cls_dir.mkdir(parents=True, exist_ok=True)
                out = cls_dir / f"{split}_{idx:05d}.jpg"
                if not out.exists():
                    img.convert("RGB").save(out)
                items.append((str(out), classes[y]))

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: eurosat | cifar10 | flowers102"
        )

    log.info("Materialized %s: %d items, %d classes", name, len(items), len(classes))
    return items, classes


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #
def stratified_split(
    items: list[tuple[str, str]], train_frac: float, val_frac: float, seed: int
) -> dict[str, list[tuple[str, str]]]:
    by_class: dict[str, list] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)

    rng = np.random.default_rng(seed)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for label, group in sorted(by_class.items()):
        idx = rng.permutation(len(group))
        n = len(group)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_test = n - n_train - n_val
        if n_test < 1:  # guarantee at least one test item per class when possible
            n_test = 1
            n_val = max(n - n_train - n_test, 0)
            n_train = max(n - n_test - n_val, 0)
        picks = [group[i] for i in idx]
        splits["train"] += picks[:n_train]
        splits["val"] += picks[n_train : n_train + n_val]
        splits["test"] += picks[n_train + n_val :]
    return splits


def build_manifest_from_items(
    items: list[tuple[str, str]],
    classes: list[str],
    root: str | Path,
    out_dir: str | Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    splits = stratified_split(items, train_frac, val_frac, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, split_items in splits.items():
        payload = {
            "dataset": Path(root).name,
            "classes": classes,
            "root": str(Path(root).resolve()),
            "items": [
                {
                    "path": os.path.relpath(p, root).replace("\\", "/"),
                    "label": label,
                    "label_idx": classes.index(label),
                }
                for p, label in split_items
            ],
        }
        p = out_dir / f"{split}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[split] = p
        log.info("Manifest %s: %d items -> %s", split, len(split_items), p)
    return paths


def create_manifests(cfg: DataConfig, seed: int = 42) -> dict[str, Path]:
    items, classes = materialize(cfg.name, cfg.root, cfg.download)
    return build_manifest_from_items(
        items,
        classes,
        Path(cfg.root) / "raw" / cfg.name,
        Path(cfg.root) / "manifests" / cfg.name,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Manifest + Dataset
# --------------------------------------------------------------------------- #
@dataclass
class Manifest:
    dataset: str
    classes: list[str]
    root: Path
    items: list[dict]
    split: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset", "classes", "root", "items"):
            if key not in payload:
                raise ValueError(f"Manifest {path} is missing required key '{key}'")
        return cls(
            dataset=payload["dataset"],
            classes=payload["classes"],
            root=Path(payload["root"]),
            items=payload["items"],
            split=path.stem,
        )


class ManifestDataset(Dataset):
    """Reads items from a manifest. transform=None returns raw PIL images
    (used by the corruption/stress benchmark, which needs pre-transform access)."""

    def __init__(self, manifest: Manifest | str | Path, transform=None):
        if not isinstance(manifest, Manifest):
            manifest = Manifest.load(manifest)
        self.manifest = manifest
        self.transform = transform
        self.paths: list[str] = [str(manifest.root / it["path"]) for it in manifest.items]
        self.labels: list[int] = [int(it["label_idx"]) for it in manifest.items]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def dataset_stats(manifest: Manifest) -> dict[str, int]:
    return dict(sorted(Counter(it["label"] for it in manifest.items).items()))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def manifest_dir_for(cfg: DataConfig) -> Path:
    return Path(cfg.root) / "manifests" / cfg.name


def build_loaders(data_cfg: DataConfig, seed: int = 42):
    """Build (train_loader, val_loader, test_loader, classes) from manifests."""
    mdir = manifest_dir_for(data_cfg)
    manifests = {}
    for split in ("train", "val", "test"):
        p = mdir / f"{split}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing manifest {p}. Run: python scripts/prepare_data.py --config <config.yaml>"
            )
        manifests[split] = Manifest.load(p)

    classes = manifests["train"].classes
    for split, m in manifests.items():
        if m.classes != classes:
            raise RuntimeError(f"Class mismatch between train and {split} manifests")

    train_tf = build_train_transform(data_cfg.img_size)
    eval_tf = build_eval_transform(data_cfg.img_size)

    train_ds = ManifestDataset(manifests["train"], train_tf)
    val_ds = ManifestDataset(manifests["val"], eval_tf)
    test_ds = ManifestDataset(manifests["test"], eval_tf)

    gen = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, classes
