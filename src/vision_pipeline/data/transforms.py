"""Train/eval image transforms (ImageNet normalization — matches both backbones)."""
from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224) -> T.Compose:
    """Light augmentation: crop + flip + mild color jitter.

    Deliberately modest — the stress benchmark measures *robustness*, so we do
    not train on the same corruptions we evaluate on.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(
                img_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BILINEAR
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
