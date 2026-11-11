"""Backbone factory + classifier wrapper.

All backbones expose the same interface: ``model.backbone(x) -> pooled features``
and ``model(x) -> logits``. The classification head is always a fresh
``nn.Linear`` (the pretrained head has the wrong output size anyway).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet50,
    vit_b_16,
)

from vision_pipeline.config import ModelConfig


class Classifier(nn.Module):
    """backbone -> pooled features -> Linear head."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate features (used by the forgetting probe)."""
        return self.backbone(x)


def _resnet_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Rebuild a ResNet as a Sequential so layer names are stable for LoRA targeting
    (e.g. ``backbone.layer1.0.conv1``) while the stem stays clearly separate."""
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        dim = 2048
    elif name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        dim = 512
    else:  # pragma: no cover
        raise ValueError(name)

    backbone = nn.Sequential(
        OrderedDict(
            [
                ("conv1", net.conv1),
                ("bn1", net.bn1),
                ("relu", net.relu),
                ("maxpool", net.maxpool),
                ("layer1", net.layer1),
                ("layer2", net.layer2),
                ("layer3", net.layer3),
                ("layer4", net.layer4),
                ("avgpool", net.avgpool),
                ("flatten", nn.Flatten(1)),
            ]
        )
    )
    return backbone, dim


def _vit_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    net = vit_b_16(weights=weights)
    net.heads = nn.Identity()  # forward now returns 768-d pre-logit features
    return net, 768


def build_classifier(cfg: ModelConfig, num_classes: int) -> Classifier:
    """Build backbone + fresh head. For linear_probe, freeze the backbone here."""
    if cfg.backbone == "vit_b_16":
        backbone, dim = _vit_backbone(cfg.pretrained)
    elif cfg.backbone in {"resnet50", "resnet18"}:
        backbone, dim = _resnet_backbone(cfg.backbone, cfg.pretrained)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    model = Classifier(backbone, dim, num_classes)

    if cfg.method == "linear_probe":
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def trainable_param_summary(model: nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 3),
    }
