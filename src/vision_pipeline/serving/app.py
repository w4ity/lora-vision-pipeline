"""FastAPI serving with an A/B comparison endpoint.

Models are loaded at startup from the LVP_MODELS environment variable:
    LVP_MODELS="lora=outputs/exp03_lora_r16_vit,linear=outputs/exp01_linear_probe_vit"
    LVP_DEVICE=auto|cuda|cpu

Endpoints:
    GET  /health                 -> status, device, loaded model labels
    POST /predict?model=lora     -> single-model prediction (multipart file)
    POST /predict/ab             -> side-by-side predictions from ALL loaded models
"""
from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from vision_pipeline.data.transforms import build_eval_transform
from vision_pipeline.evaluation.evaluate import load_trained
from vision_pipeline.models.factory import resolve_device

_STATE: dict[str, Any] = {"models": {}, "device": "cpu"}


class TopKItem(BaseModel):
    label: str
    prob: float


class Prediction(BaseModel):
    model: str
    pred: str
    probs: list[TopKItem]
    latency_ms: float
    img_size: int


class ABResponse(BaseModel):
    results: list[Prediction]
    agreement: bool


def _parse_models(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in filter(None, spec.split(",")):
        label, _, path = part.partition("=")
        if label and path:
            out[label.strip()] = path.strip()
    return out


def _load_models(spec: str, device: torch.device) -> dict[str, dict]:
    models = {}
    for label, run_dir in _parse_models(spec).items():
        model, classes, cfg, _meta = load_trained(run_dir, device)
        models[label] = {
            "model": model,
            "classes": classes,
            "transform": build_eval_transform(cfg.data.img_size),
            "img_size": cfg.data.img_size,
        }
    return models


@asynccontextmanager
async def lifespan(_app: FastAPI):
    device = resolve_device(os.environ.get("LVP_DEVICE", "auto"))
    spec = os.environ.get("LVP_MODELS", "")
    _STATE["models"] = _load_models(spec, device) if spec else {}
    _STATE["device"] = str(device)
    yield
    _STATE["models"] = {}


app = FastAPI(
    title="LoRA Vision A/B API",
    description="Compare base (linear probe) vs LoRA fine-tuned models side by side.",
    version="0.1.0",
    lifespan=lifespan,
)


def _load_image(file: UploadFile) -> Image.Image:
    data = file.file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file") from None


def _predict_one(entry: dict, img: Image.Image, top_k: int, label: str) -> Prediction:
    t0 = time.perf_counter()
    x = entry["transform"](img).unsqueeze(0).to(_STATE["device"])
    with torch.inference_mode():
        logits = entry["model"](x)
        probs = torch.softmax(logits[0], dim=-1)
    k = min(top_k, probs.numel())
    top = torch.topk(probs, k)
    latency = (time.perf_counter() - t0) * 1000
    items = [
        TopKItem(label=entry["classes"][int(i)], prob=round(float(p), 5))
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    return Prediction(
        model=label, pred=items[0].label, probs=items,
        latency_ms=round(latency, 2), img_size=entry["img_size"],
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": _STATE["device"],
        "models": sorted(_STATE["models"]),
    }


@app.post("/predict", response_model=Prediction)
def predict(
    file: UploadFile = File(...),
    model: str | None = Query(default=None, description="Model label to use"),
    top_k: int = Query(default=5, ge=1, le=50),
) -> Prediction:
    if not _STATE["models"]:
        raise HTTPException(status_code=503, detail="No models loaded. Set LVP_MODELS.")
    label = model or next(iter(_STATE["models"]))
    if label not in _STATE["models"]:
        raise HTTPException(status_code=404, detail=f"Unknown model '{label}'")
    img = _load_image(file)
    return _predict_one(_STATE["models"][label], img, top_k, label)


@app.post("/predict/ab", response_model=ABResponse)
def predict_ab(file: UploadFile = File(...), top_k: int = Query(default=5, ge=1, le=50)) -> ABResponse:
    if not _STATE["models"]:
        raise HTTPException(status_code=503, detail="No models loaded. Set LVP_MODELS.")
    img = _load_image(file)
    results = [
        _predict_one(entry, img, top_k, label)
        for label, entry in _STATE["models"].items()
    ]
    agreement = len({r.pred for r in results}) == 1
    return ABResponse(results=results, agreement=agreement)
