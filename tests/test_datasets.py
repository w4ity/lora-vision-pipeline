"""Manifests: stratified proportions, dataset wrapper, stats."""
import torch

from vision_pipeline.data.datasets import (
    Manifest,
    ManifestDataset,
    build_manifest_from_items,
    dataset_stats,
)
from vision_pipeline.data.transforms import build_eval_transform


def test_split_proportions(manifests):
    m = Manifest.load(manifests["train"])
    counts = dataset_stats(m)
    # 8 per class, 0.75 train -> 6 per class x 3 classes = 18
    assert len(m.items) == 18
    assert all(v == 6 for v in counts.values())

    val = Manifest.load(manifests["val"])
    test = Manifest.load(manifests["test"])
    assert len(val.items) == 3 and len(test.items) == 3
    # no overlap between splits
    train_paths = {i["path"] for i in m.items}
    val_paths = {i["path"] for i in val.items}
    test_paths = {i["path"] for i in test.items}
    assert not (train_paths & val_paths) and not (train_paths & test_paths) and not (val_paths & test_paths)


def test_deterministic_splits(synthetic_items, tmp_path):
    root, items = synthetic_items
    a = build_manifest_from_items(items, ["alpha", "beta", "gamma"], root, tmp_path / "a", 0.75, 0.125, seed=0)
    b = build_manifest_from_items(items, ["alpha", "beta", "gamma"], root, tmp_path / "b", 0.75, 0.125, seed=0)
    pa = {i["path"] for i in Manifest.load(a["train"]).items}
    pb = {i["path"] for i in Manifest.load(b["train"]).items}
    assert pa == pb  # same seed -> identical split


def test_manifest_dataset(manifests):
    ds = ManifestDataset(manifests["test"], build_eval_transform(64))
    assert len(ds) == 3
    img, y = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= y < 3
    # raw PIL mode (transform=None) is what the corruption layer consumes
    raw = ManifestDataset(manifests["test"], transform=None)
    pil, y2 = raw[0]
    assert pil.size[0] == 48 and isinstance(y2, int)
