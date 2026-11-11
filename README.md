# LoRA Vision Fine-Tuning Pipeline

This project fine-tunes open-source vision models with LoRA on a domain-specific dataset and measures the effect rigorously. It trains a linear-probe baseline and several LoRA configurations on EuroSAT satellite imagery, compares them on a held-out test set, runs robustness and catastrophic-forgetting checks, and serves the final models behind a small A/B API.

The main result: test macro-F1 improves from 96.44% (frozen backbone) to 98.78% (LoRA rank 32) while training only 3.3% of the parameters. Every number in this README comes from artifacts in `reports/` and `outputs/`.

![Validation macro-F1 across the sweep](docs/screenshots/mlflow_compare_curves.png)

The five runs in one view. The frozen backbone plateaus around 0.96 and every LoRA variant reaches about 0.987 within three epochs.

## What the pipeline does

1. Prepares EuroSAT (27,000 Sentinel-2 tiles, 10 classes) and writes frozen train/val/test manifests (80/10/10, stratified, seeded).
2. Trains a linear-probe baseline and LoRA runs (ranks 8, 16, 32 on ViT-B/16, plus rank 16 on ResNet-50) with early stopping, checkpointing and experiment tracking.
3. Evaluates every run on the test split, on a corruption stress benchmark, and with a forgetting probe on CIFAR-10.
4. Compares baseline against fine-tuned per class and per example, with a paired exact binomial test.
5. Serves any trained model behind a FastAPI endpoint with an A/B comparison view.

Each stage is a single command, and each training run is fully described by one YAML file in `configs/`.

| Stage | Implementation |
|---|---|
| Data | `scripts/prepare_data.py`: download, materialize, stratified manifests |
| Training | `vision_pipeline/training/`: LoRA via PEFT, dual learning rate, warmup and cosine schedule, AMP, early stopping, MLflow and CSV logging |
| Evaluation | `scripts/evaluate.py` (test plus stress), `scripts/compare.py` (head-to-head plus binomial test), `scripts/run_forgetting.py` (feature drift and kNN) |
| Serving | `vision_pipeline/serving/`: FastAPI with `/predict` and `/predict/ab`, latency benchmark |
| Packaging | `Dockerfile`, `docker-compose.yml`, `reports/EXPERIMENT_REPORT.md` |

## Architecture

```
                    +------------------------------+
                    |        configs/*.yaml        |
                    |  one YAML = one repeatable run |
                    +---------------+--------------+
                                    |
  data/raw/<ds> -> stratified splits |
  (materialized)  train/val/test.json|
                  frozen manifests   v
                       +-------------------------+
                       |     Trainer (PEFT)      |
                       |  ViT-B/16 + LoRA (FFN)  |
                       |  ResNet50 + LoRA (1x1)  |
                       |  or linear probe        |
                       +------------+------------+
                                    | best.pt + adapter/ + meta.json
                 +------------------+------------------+
                 v                  v                  v
          evaluate.py          compare.py       serve.py (FastAPI)
        clean + stress      base vs LoRA        /predict  /predict/ab
                            binomial test
                 +------------------+------------------+
                                    v
              forgetting.py (feature drift + kNN on CIFAR-10)
```

## Getting started

Requires Python 3.11 or newer. A GPU is optional; ResNet trains fine on CPU and ViT is slower there.

```bash
# 1. Environment (Windows Git Bash shown; on Linux/macOS use .venv/bin/activate)
python -m venv .venv
source .venv/Scripts/activate

# NVIDIA GPU on Windows: install CUDA wheels first (PyPI default is CPU-only there)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# everything else
pip install -r requirements.txt
pip install -e .

# sanity check, should print True
python -c "import torch; print(torch.cuda.is_available())"

# 2. Dataset: download EuroSAT and build the frozen manifests
python scripts/prepare_data.py --config configs/base.yaml

# 3. Train the baseline, then the best LoRA run
python scripts/train.py --config configs/experiments/exp01_linear_probe_vit.yaml
python scripts/train.py --config configs/experiments/exp04_lora_r32_vit.yaml

# 4. Or run the full sweep (ranks 8/16/32 + ResNet)
python scripts/sweep.py --configs exp01_linear_probe_vit exp02_lora_r8_vit exp03_lora_r16_vit exp04_lora_r32_vit exp05_lora_r16_resnet

# 5. Evaluation stack
python scripts/evaluate.py --run-dir outputs/exp04_lora_r32_vit
python scripts/compare.py --run-a outputs/exp01_linear_probe_vit --run-b outputs/exp04_lora_r32_vit
python scripts/run_forgetting.py --run-dir outputs/exp04_lora_r32_vit

# 6. Inference benchmark
python scripts/benchmark_inference.py --run-dir outputs/exp04_lora_r32_vit

# 7. A/B demo API, then POST an image to /predict/ab
python scripts/serve.py --models "lora_r32=outputs/exp04_lora_r32_vit,linear_probe=outputs/exp01_linear_probe_vit"

# Tests (offline, CPU, no downloads)
python -m pytest -v
```

## Results

All metrics below come from the shared 2,700-image test split. Validation was used for model selection only.

| Config | Trainable params | val macro-F1 | test acc | test macro-F1 | train (min) |
|---|---|---|---|---|---|
| exp01 linear probe (ViT, baseline) | 7,690 (0.009%) | 96.26% | 96.56% | 96.44% | 8.9 |
| exp02 LoRA r8 (ViT) | 744,970 (0.86%) | 98.69% | 98.70% | 98.65% | 19.6 |
| exp03 LoRA r16 (ViT) | 1,482,250 (1.70%) | 98.65% | 98.48% | 98.44% | 19.4 |
| exp04 LoRA r32 (ViT) | 2,956,810 (3.33%) | 98.69% | 98.81% | 98.78% | 19.6 |
| exp05 LoRA r16 (ResNet-50) | 592,906 (2.46%) | 98.67% | 98.26% | 98.19% | 10.4 |

![Confusion matrices: baseline vs LoRA](docs/screenshots/fig_confusion_matrices.png)

Test errors drop from 93 to 32. The remaining gains concentrate in the classes the baseline confused most, such as PermanentCrop and AnnualCrop.

![Per-class test F1](docs/screenshots/fig_per_class_f1.png)

LoRA improves every class. The largest gains are PermanentCrop (+4.3 F1), River (+4.3) and Highway (+3.5).

### What the numbers say

- LoRA beats the frozen baseline by +2.34 F1 points. On the same 2,700 ordered test images it fixes 73 baseline errors and introduces 12, and the paired exact binomial test gives p below 1e-12, so the difference is not chance.
- The result is not sensitive to rank. Rank 8 already reaches 98.65% test F1 with 0.9% of the parameters trained.
- The same LoRA idea transfers to ResNet-50, which reaches 98.19% in about half the training time.
- Robustness improves only slightly on average (75.05% to 75.91% mean accuracy across the stress benchmark). LoRA handles blur, JPEG, brightness and contrast better at every severity, and handles strong gaussian noise slightly worse. Fine-tuning is not automatically more robust.

![Stress benchmark](docs/screenshots/fig_stress.png)

- The forgetting probe measures the cost of adaptation. kNN accuracy on CIFAR-10 (a domain disjoint from EuroSAT) drops 6.7, 4.2 and 5.4 points for ranks 8, 16 and 32. The drop does not scale with rank, so the FFN adaptation itself, not its capacity, drives the drift.

![Catastrophic-forgetting probe](docs/screenshots/fig_forgetting.png)

The A/B endpoint makes the difference visible on single images. On the test tile below, the linear probe answers PermanentCrop with 71.8% confidence and the LoRA model answers AnnualCrop with 95.8%.

![A/B demo](docs/screenshots/ab_demo.png)

`outputs/` and trained weights are gitignored. Everything regenerates from the commands above, and `reports/EXPERIMENT_REPORT.md` documents each claim with its artifact path.

## Implementation notes

1. Baseline choice. A pretrained model with a random head scores at chance level, so the honest reference is a linear probe: frozen backbone plus a trained head. LoRA is then the only treatment, and `compare.py` quantifies the delta.
2. Dead adapters on torchvision ViT. The first sweep produced three LoRA runs identical to the baseline to 16 decimal places. The cause: torchvision's ViT uses `nn.MultiheadAttention`, and `F.multi_head_attention_forward` applies the output projection on raw weight tensors without ever calling `out_proj.forward()`, so a LoRA wrapper there gets zero gradient. The fix targets the MLP projections `mlp.0` and `mlp.3` of every encoder layer, and a regression test now asserts that adapters run in the forward path and receive gradients.
3. BatchNorm trap. Frozen BatchNorm layers still update running statistics in `train()` mode even with `requires_grad=False`. `configure_train_mode` forces every non-adapted module to `eval()` during LoRA training, which keeps the backbone from drifting silently.
4. Saving the head. The fresh classification head must train and travel with the adapter, so it is listed in `modules_to_save`.
5. Data governance. Splits are written as JSON manifests before any training. The test manifest is only read by final-reporting scripts. Each run directory stores its resolved config in `meta.json` plus a config hash.
6. Stress benchmark. Five corruptions at five severities, applied at evaluation time only, measure whether the model improved on data it never saw in augmentation.
7. Forgetting probe. Penultimate features are compared between the original pretrained backbone and the adapted one on CIFAR-10, using paired cosine distance and kNN accuracy.
8. Small deployment artifact. The LoRA adapter including the head is 12 MB next to a 331 MB base model, so one base can host many domain adapters.

## Configuration

Main fields of `configs/base.yaml`. Experiment files inherit from it with `inherit`.

| Field | Default | Meaning |
|---|---|---|
| `model.backbone` | `vit_b_16` | `vit_b_16`, `resnet50` or `resnet18` |
| `model.method` | `lora` | `lora` or `linear_probe` (baseline) |
| `lora.rank` / `alpha` | 16 / 32 | adapter capacity |
| `train.lr` / `train.head_lr` | 2.0e-4 / 1.0e-3 | adapters vs fresh head |
| `train.epochs` | 5 | with early stopping, patience 3, gated by val macro-F1 |
| `data.name` | `eurosat` | `eurosat`, `cifar10` or `flowers102` |
| `tracking.backend` | `mlflow` | SQLite store (`sqlite:///mlflow.db`), `none` to disable |

EuroSAT caveat, documented on purpose: its tiles come from overlapping Sentinel-2 scenes, so random splits can share scenes between train and test. This is the standard protocol for comparability; a scene-aware split is listed in the roadmap.

## Repo layout

```
configs/            base.yaml + 5 experiment YAMLs (inheritance)
src/vision_pipeline/
  config.py         typed YAML config, inheritance, validation
  data/             dataset materialization, manifests, transforms
  models/           backbone factory, classifier, LoRA wiring
  training/         metrics, tracker, trainer, run, sweep
  evaluation/       corruptions, evaluate, compare, forgetting probe
  serving/          FastAPI app
scripts/            CLI entry points
tests/              offline pytest suite with an end-to-end training smoke test
reports/            experiment report
docs/screenshots/   figures used in this README
```

## Requirements

- Python 3.11 or newer, about 4 GB of disk for weights and datasets
- Any NVIDIA GPU with 4 GB or more makes ViT training minutes instead of an hour; everything falls back to CPU
- First run downloads: EuroSAT (~90 MB), ViT-B/16 weights (~340 MB), ResNet-50 weights (~100 MB), CIFAR-10 (~170 MB)

## Tech stack

PyTorch, torchvision, Hugging Face PEFT, MLflow, FastAPI, pytest, Docker.
