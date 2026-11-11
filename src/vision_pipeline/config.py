"""Typed, YAML-driven configuration with inheritance.

Every experiment is a single YAML file. `inherit: base.yaml` merges the parent
deeply, so an experiment file only contains what makes it different. Configs are
the unit of reproducibility: a run directory stores the exact resolved config
that produced it (`meta.json`), and `config_hash()` fingerprints it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    name: str = "eurosat"
    root: str = "data"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_frac: float = 0.8
    val_frac: float = 0.1
    pin_memory: bool = True
    download: bool = True


@dataclass
class ModelConfig:
    backbone: str = "vit_b_16"      # vit_b_16 | resnet50 | resnet18
    pretrained: bool = True
    method: str = "lora"            # lora | linear_probe


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | None = None  # None -> backbone default in models/lora.py


@dataclass
class TrainConfig:
    lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_epochs: float = 0.5
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 3
    max_steps_per_epoch: int | None = None


@dataclass
class EvalConfig:
    batch_size: int = 64
    corruption_severities: list[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class TrackingConfig:
    backend: str = "mlflow"         # mlflow | none
    tracking_uri: str = "mlruns"
    experiment_name: str = "lora-vision"


@dataclass
class ExperimentConfig:
    name: str = "exp"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig | None = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def flat_params(self) -> dict[str, Any]:
        """Flatten nested sections into MLflow-friendly params."""
        out: dict[str, Any] = {}

        def _walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, list):
                out[prefix] = json.dumps(obj)
            else:
                out[prefix] = obj

        _walk("", self.to_dict())
        return out

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _build(cls: type, data: dict[str, Any] | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys with a warning."""
    if data is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        log.warning("Ignoring unknown config keys for %s: %s", cls.__name__, unknown)
    return cls(**{k: v for k, v in data.items() if k in known})


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("inherit"):
        parent = (path.parent / raw["inherit"]).resolve()
        parent_raw = yaml.safe_load(parent.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(parent_raw, raw)
    raw.pop("inherit", None)

    cfg = ExperimentConfig(
        name=raw.get("name", "exp"),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        output_dir=str(raw.get("output_dir", "outputs")),
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        lora=None if raw.get("lora") is None else _build(LoraConfig, raw.get("lora")),
        train=_coerce_train_floats(_build(TrainConfig, raw.get("train"))),
        eval=_build(EvalConfig, raw.get("eval")),
        tracking=_build(TrackingConfig, raw.get("tracking")),
    )
    _validate(cfg)
    return cfg


def config_from_dict(d: dict[str, Any]) -> ExperimentConfig:
    """Inverse of `ExperimentConfig.to_dict()` — used to reload runs from meta.json."""
    cfg = ExperimentConfig(
        name=d.get("name", "exp"),
        seed=int(d.get("seed", 42)),
        device=str(d.get("device", "auto")),
        output_dir=str(d.get("output_dir", "outputs")),
        data=_build(DataConfig, d.get("data")),
        model=_build(ModelConfig, d.get("model")),
        lora=None if d.get("lora") is None else _build(LoraConfig, d.get("lora")),
        train=_coerce_train_floats(_build(TrainConfig, d.get("train"))),
        eval=_build(EvalConfig, d.get("eval")),
        tracking=_build(TrackingConfig, d.get("tracking")),
    )
    return cfg


def _coerce_train_floats(train: TrainConfig) -> TrainConfig:
    """YAML 1.1 parses '2e-4' (no decimal point) as a string, not a float.
    Coerce the known numeric fields so a hand-edited config can't produce a
    cryptic failure deep inside the optimizer."""
    for f in ("lr", "head_lr", "weight_decay", "label_smoothing", "grad_clip", "warmup_epochs"):
        v = getattr(train, f)
        if isinstance(v, str):
            setattr(train, f, float(v))
    return train


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.model.backbone not in {"vit_b_16", "resnet50", "resnet18"}:
        raise ValueError(f"Unknown backbone: {cfg.model.backbone}")
    if cfg.model.method not in {"lora", "linear_probe"}:
        raise ValueError(f"Unknown method: {cfg.model.method}")
    if not 0.0 < cfg.data.train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {cfg.data.train_frac}")
    if not 0.0 <= cfg.data.val_frac < 1.0 - cfg.data.train_frac:
        raise ValueError(f"val_frac {cfg.data.val_frac} leaves no test split")
    if cfg.model.method == "lora" and cfg.lora is None:
        raise ValueError("method=lora requires a `lora:` section")
    if cfg.lora is not None and cfg.lora.rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {cfg.lora.rank}")
