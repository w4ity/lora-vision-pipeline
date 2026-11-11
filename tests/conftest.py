"""Shared fixtures: a tiny synthetic image dataset written to disk + manifests.

No downloads, no pretrained weights — the whole suite runs offline on CPU.
"""
import sys
from pathlib import Path

# Make the package importable without requiring `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from vision_pipeline.data.datasets import build_manifest_from_items

CLASSES = ["alpha", "beta", "gamma"]


@pytest.fixture(scope="session")
def synthetic_items(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    items = []
    for ci, cls in enumerate(CLASSES):
        for i in range(8):  # 8 images per class
            arr = np.zeros((48, 48, 3), dtype=np.uint8)
            arr[..., ci] = rng.integers(120, 255, arr[..., ci].shape)  # class-coded channel
            p = root / cls / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p)
            items.append((str(p), cls))
    return root, items


@pytest.fixture
def manifests(synthetic_items, tmp_path):
    root, items = synthetic_items
    return build_manifest_from_items(
        items, CLASSES, root, tmp_path / "manifests",
        train_frac=0.75, val_frac=0.125, seed=0,
    )
