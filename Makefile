# Convenience targets. On Windows (Git Bash) use `make <target>` or run the
# underlying commands directly — every command is also documented in README.md.

PY ?= python
CONFIG ?= configs/experiments/exp03_lora_r16_vit.yaml
RUN ?= exp03_lora_r16_vit
BASELINE ?= exp01_linear_probe_vit

.PHONY: setup data train sweep eval compare forgetting benchmark serve test docker-build docker-up

setup:  ## create venv and install deps (Windows/Git Bash path)
	$(PY) -m venv .venv
	. .venv/Scripts/activate && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt

data:  ## download dataset, materialize files, build stratified manifests
	$(PY) scripts/prepare_data.py --config $(CONFIG)

train:  ## train a single experiment
	$(PY) scripts/train.py --config $(CONFIG)

sweep:  ## train the full sweep (baseline + LoRA ranks + ResNet)
	$(PY) scripts/sweep.py --configs exp01_linear_probe_vit exp02_lora_r8_vit exp03_lora_r16_vit exp04_lora_r32_vit exp05_lora_r16_resnet

eval:  ## evaluate a run on the test split + stress benchmark
	$(PY) scripts/evaluate.py --run-dir outputs/$(RUN)

compare:  ## head-to-head: linear probe (baseline) vs LoRA
	$(PY) scripts/compare.py --run-a outputs/$(BASELINE) --run-b outputs/$(RUN)

forgetting:  ## catastrophic-forgetting probe (feature drift + kNN on CIFAR-10)
	$(PY) scripts/run_forgetting.py --run-dir outputs/$(RUN)

benchmark:  ## inference throughput + latency P50/P95
	$(PY) scripts/benchmark_inference.py --run-dir outputs/$(RUN)

serve:  ## local A/B API (set MODELS to override)
	$(PY) scripts/serve.py --models "lora=outputs/$(RUN),linear=outputs/$(BASELINE)"

test:
	$(PY) -m pytest -v

docker-build:
	docker build -t lora-vision-api .

docker-up:
	docker compose up --build
