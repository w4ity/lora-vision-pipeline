"""LoRA wiring: exact target counts, frozen stem, trainable head, BN eval mode."""
import torch

from vision_pipeline.config import LoraConfig, ModelConfig
from vision_pipeline.models.factory import build_classifier, trainable_param_summary
from vision_pipeline.models.lora import apply_lora, configure_train_mode


def _lora_modules(peft_model):
    return [n for n, _ in peft_model.named_modules() if n.endswith(".lora_A.default")]


def test_resnet18_lora_targets_and_counts():
    base = build_classifier(ModelConfig("resnet18", pretrained=False, method="lora"), num_classes=4)
    peft_model = apply_lora(base, LoraConfig(rank=4, alpha=8), "resnet18")

    lora_mods = _lora_modules(peft_model)
    # ResNet-18 BasicBlocks have conv1/conv2 only (no conv3) -> 8 conv1 targets.
    assert len(lora_mods) == 8
    assert all(n.endswith(".conv1.lora_A.default") for n in lora_mods)
    # stem is NOT adapted
    assert not any(".conv1.lora" in n and "layer" not in n for n in lora_mods)
    # downsample projections are NOT adapted
    assert not any("downsample" in n for n in lora_mods)
    # the fresh head is saved via modules_to_save
    saved = [n for n, m in peft_model.named_modules() if "modules_to_save" in n]
    assert any("head" in n for n in saved)

    out = peft_model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)

    summary = trainable_param_summary(peft_model)
    assert summary["trainable_pct"] < 5.0  # LoRA keeps the trainable fraction tiny


def test_vit_lora_targets():
    base = build_classifier(ModelConfig("vit_b_16", pretrained=False, method="lora"), num_classes=4)
    peft_model = apply_lora(base, LoraConfig(rank=4, alpha=8), "vit_b_16")

    lora_mods = _lora_modules(peft_model)
    # 12 encoder layers x 2 MLP projections (mlp.0, mlp.3) = 24 targets.
    assert len(lora_mods) == 24
    assert all(n.endswith((".mlp.0.lora_A.default", ".mlp.3.lora_A.default")) for n in lora_mods)


def test_lora_adapters_are_in_the_forward_path():
    """Regression test born from a live failure: torchvision's
    nn.MultiheadAttention applies out_proj via F.linear on the raw weight tensor,
    never calling the module's forward — so LoRA on out_proj produced exactly-zero
    gradients and three sweep runs trained identically to the linear-probe
    baseline. Adapters must actually participate in forward AND backward."""
    for backbone, size in (("vit_b_16", 224), ("resnet18", 64)):
        base = build_classifier(ModelConfig(backbone, pretrained=False, method="lora"), num_classes=4)
        peft_model = apply_lora(base, LoraConfig(rank=2, alpha=4), backbone)

        first_lora_a = next(
            m for n, m in peft_model.named_modules() if n.endswith(".lora_A.default")
        )
        fired = []
        first_lora_a.register_forward_hook(lambda *a: fired.append(1))

        out = peft_model(torch.randn(2, 3, size, size))
        out.sum().backward()

        assert fired, f"{backbone}: LoRA wrapper never executed during forward"
        b_grads = [p.grad for n, p in peft_model.named_parameters()
                   if "lora_B" in n and p.grad is not None]
        total = sum(float(g.abs().sum()) for g in b_grads)
        assert b_grads and total > 0.0, f"{backbone}: lora_B received no gradient"

    out = peft_model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_linear_probe_freezes_backbone():
    model = build_classifier(ModelConfig("resnet18", pretrained=False, method="linear_probe"), num_classes=4)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert model.head.weight.requires_grad


def test_configure_train_mode_keeps_bn_frozen():
    base = build_classifier(ModelConfig("resnet18", pretrained=False, method="lora"), num_classes=4)
    peft_model = apply_lora(base, LoraConfig(rank=2, alpha=4), "resnet18")

    configure_train_mode(peft_model, "lora")
    # BN inside the frozen backbone must stay in eval mode (running stats frozen)
    bn = peft_model.base_model.model.backbone.bn1
    assert bn.training is False
    # the saved head copy (modules_to_save) must be in train mode
    saved_heads = [m for n, m in peft_model.named_modules()
                   if n.endswith(".head.modules_to_save.default")]
    assert saved_heads and saved_heads[0].training is True
