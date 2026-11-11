# Inference / demo image (CPU). Training runs on the host, where the GPU lives.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only wheels keep the image small.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "vision_pipeline.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
