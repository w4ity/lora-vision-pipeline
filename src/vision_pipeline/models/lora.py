"""LoRA adaptation via PEFT, with backbone-specific target selection.

Target modules (the interesting design decision — verified against the actual
installed libraries by unit tests AND a live-sweep post-mortem):
* ViT-B/16 -> the MLP projections ``mlp.0`` / ``mlp.3`` of every encoder layer
  (24 adapters) — the FFN, the torchvision analogue of the fc1/fc2 LoRA targets
  used for Hugging Face ViTs. Why not attention? torchvision's ViT uses
  ``nn.MultiheadAttention``, and ``F.multi_head_attention_forward`` applies the
  output projection via ``linear(attn_output, out_proj.weight, out_proj.bias)``
  on the raw TENSORS — it never calls ``out_proj.forward()``. A LoRA wrapper on
  ``out_proj`` is therefore dead code: forward hooks never fire and lora_B
  gradients are exactly zero (the first sweep trained three LoRA runs identical
  to the linear-probe baseline to 16 decimal places; see
  ``test_lora_adapters_are_in_the_forward_path``).
* ResNet   -> the **1x1 convs of every Bottleneck block** (``conv1`` and ``conv3``)
  for ResNet-50 -> 32 adapters. ResNet-18's BasicBlock has no ``conv3``, so only
  its 8 ``conv1`` entries are adapted. The stem (``backbone.conv1``) and the
  ``downsample`` projections stay frozen, enforced by the regex below.

Gotchas handled here (all verified by unit tests):
* PEFT matches target regexes with ``re.fullmatch`` on the full module path, so
  patterns must cover the whole path — ``conv1`` alone would match the stem.
* ``modules_to_save=["head"]`` keeps the fresh classification head trainable and
  included in ``save_pretrained`` output.
* BatchNorm: frozen BN layers still update running statistics in ``train()``
  mode even with ``requires_grad=False``. ``configure_train_mode`` puts every
  non-adapted module back in ``eval()`` after each ``model.train()`` call to
  prevent silent backbone drift during LoRA training.
"""
from __future__ import annotations

import logging

import torch.nn as nn

from vision_pipeline.config import LoraConfig

log = logging.getLogger(__name__)

DEFAULT_LORA_TARGETS: dict[str, str] = {
    "vit_b_16": r".*\.mlp\.(0|3)$",
    "resnet50": r".*layer\d+\.\d+\.conv[13]$",
    "resnet18": r".*layer\d+\.\d+\.conv[13]$",
}


def is_peft_model(model: nn.Module) -> bool:
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:  # pragma: no cover
        return False


def apply_lora(model: nn.Module, lora_cfg: LoraConfig, backbone_name: str) -> nn.Module:
    """Wrap a Classifier with LoRA adapters. Returns a PEFT model whose forward
    signature is unchanged (logits in, logits out)."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    targets = lora_cfg.target_modules or DEFAULT_LORA_TARGETS[backbone_name]
    peft_cfg = PeftLoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=targets,
        modules_to_save=["head"],  # the fresh classification head must train + be saved
        bias="none",
    )
    peft_model = get_peft_model(model, peft_cfg)
    configure_train_mode(peft_model, "lora")
    return peft_model


def set_non_adapted_to_eval(model: nn.Module) -> None:
    """Everything that is not a LoRA adapter or a modules_to_save head -> eval().
    Critical for ResNet: BN running stats must not move while training adapters."""
    for name, module in model.named_modules():
        if "lora_" in name or "modules_to_save" in name:
            module.train()
        else:
            module.eval()


def configure_train_mode(model: nn.Module, method: str) -> None:
    """Call after every ``model.train()`` to keep frozen modules in eval mode."""
    model.train()
    if is_peft_model(model):
        set_non_adapted_to_eval(model)
    elif method == "linear_probe":
        model.backbone.eval()
