"""Corruption benchmark: bounds, determinism, dataset wrapper."""
import numpy as np
import torch
from PIL import Image

from vision_pipeline.evaluation.corrupt import CORRUPTIONS, CorruptedDataset, apply_corruption
from vision_pipeline.data.datasets import Manifest, ManifestDataset
from vision_pipeline.data.transforms import build_eval_transform


def test_all_corruptions_apply():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for name in CORRUPTIONS:
        for sev in (1, 5):
            out = apply_corruption(img, name, sev, seed=0)
            assert out.size == img.size
            assert out.mode == "RGB"


def test_invalid_inputs():
    img = Image.new("RGB", (8, 8))
    try:
        apply_corruption(img, "nope", 1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        apply_corruption(img, "jpeg", 6)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_corrupted_dataset_wrapper(manifests):
    raw = ManifestDataset(Manifest.load(manifests["test"]), transform=None)
    ds = CorruptedDataset(raw, "jpeg", 3, build_eval_transform(64), seed=0)
    assert len(ds) == len(raw)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    # deterministic per index
    x2, _ = ds[0]
    assert torch.equal(x, x2)
