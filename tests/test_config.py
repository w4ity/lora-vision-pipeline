"""Config loading: inheritance, unknown-key tolerance, round-trip."""
import textwrap

from vision_pipeline.config import config_from_dict, load_config


def test_load_with_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(textwrap.dedent("""
        name: base
        seed: 7
        data: {name: eurosat, img_size: 224, batch_size: 32}
        model: {backbone: vit_b_16, method: lora}
        lora: {rank: 16, alpha: 32}
    """), encoding="utf-8")
    (tmp_path / "child.yaml").write_text(textwrap.dedent("""
        inherit: base.yaml
        name: child
        lora: {rank: 8}
    """), encoding="utf-8")

    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "child"
    assert cfg.seed == 7                       # inherited
    assert cfg.lora.rank == 8                  # overridden
    assert cfg.lora.alpha == 32                # inherited within the lora section
    assert cfg.model.backbone == "vit_b_16"    # deep-merged section untouched


def test_lora_null_disables(tmp_path):
    (tmp_path / "exp.yaml").write_text(textwrap.dedent("""
        name: exp
        model: {backbone: resnet50, method: linear_probe}
        lora: null
    """), encoding="utf-8")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.lora is None


def test_unknown_keys_ignored(tmp_path):
    (tmp_path / "exp.yaml").write_text(
        "name: exp\nmodel: {method: linear_probe}\nfuture_option: 42\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.name == "exp"


def test_roundtrip():
    cfg = load_config("configs/base.yaml")
    cfg2 = config_from_dict(cfg.to_dict())
    assert cfg == cfg2
    assert cfg.config_hash() == cfg2.config_hash()
    assert any(k.startswith("data.") for k in cfg.flat_params())
