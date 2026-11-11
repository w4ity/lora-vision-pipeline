"""A miniature ImageNet-C: five corruptions x five severities, applied to the
test manifest at evaluation time. It measures whether fine-tuning actually
improved *robustness* or just clean-split accuracy.

Corruptions deliberately do NOT appear in the training augmentation: we report
honest out-of-distribution behavior, not memorized robustness.
"""
from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

# severity 1..5 -> parameter values (index severity-1)
_SEVERITY: dict[str, list[float]] = {
    "gaussian_noise": [0.02, 0.04, 0.08, 0.12, 0.18],  # std as fraction of 255
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 3.0],        # radius px
    "jpeg": [92, 80, 65, 50, 35],                       # quality
    "brightness": [0.85, 0.75, 0.65, 0.55, 0.40],       # factor
    "contrast": [0.85, 0.70, 0.60, 0.50, 0.35],         # factor
}

CORRUPTIONS: list[str] = list(_SEVERITY)


def apply_corruption(img: Image.Image, name: str, severity: int, seed: int | None = None) -> Image.Image:
    if name not in _SEVERITY:
        raise ValueError(f"Unknown corruption '{name}'. Supported: {CORRUPTIONS}")
    if not 1 <= severity <= 5:
        raise ValueError(f"Severity must be 1..5, got {severity}")

    img = img.convert("RGB")
    if name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.float32)
        sigma = 255.0 * _SEVERITY[name][severity - 1]
        arr = arr + rng.normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif name == "gaussian_blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=_SEVERITY[name][severity - 1]))
    elif name == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(_SEVERITY[name][severity - 1]))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    elif name == "brightness":
        img = ImageEnhance.Brightness(img).enhance(_SEVERITY[name][severity - 1])
    elif name == "contrast":
        img = ImageEnhance.Contrast(img).enhance(_SEVERITY[name][severity - 1])
    return img


class CorruptedDataset(Dataset):
    """Wraps a manifest dataset (transform=None -> raw PIL) and applies one
    corruption deterministically per index, followed by the eval transform."""

    def __init__(self, base: Dataset, corruption: str, severity: int, transform: Callable, seed: int = 0):
        if corruption not in _SEVERITY:
            raise ValueError(f"Unknown corruption '{corruption}'")
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be 1..5, got {severity}")
        self.base = base
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, y = self.base[i]  # PIL image, label
        img = apply_corruption(img, self.corruption, self.severity, seed=self.seed + i)
        return self.transform(img), y
