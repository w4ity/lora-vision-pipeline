"""Inference benchmark for a trained run: throughput (imgs/s at batch 1/8/32)
and single-image latency (P50/P95)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.evaluation.evaluate import load_trained
from vision_pipeline.models.factory import resolve_device
from vision_pipeline.utils import get_logger, md_table

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=100)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, classes, cfg, meta = load_trained(args.run_dir, device)
    size = cfg.data.img_size
    name = cfg.name

    results = {"run": name, "device": str(device), "img_size": size, "throughput": [], "latency": {}}

    # ---- throughput ---- #
    for batch_size in (1, 8, 32):
        x = torch.randn(batch_size, 3, size, size, device=device)
        with torch.inference_mode():
            for _ in range(10):  # warmup
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        imgs_per_s = batch_size * args.iters / elapsed
        results["throughput"].append({"batch": batch_size, "imgs_per_s": round(imgs_per_s, 1)})
        log.info("batch=%d -> %.1f imgs/s", batch_size, imgs_per_s)

    # ---- single-image latency ---- #
    x = torch.randn(1, 3, size, size, device=device)
    times = []
    with torch.inference_mode():
        for _ in range(args.latency_iters):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    results["latency"] = {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }

    out_dir = Path("outputs") / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / f"{name}_benchmark.md").write_text(
        f"# Inference benchmark: {name}\n\n"
        + md_table(
            ["batch", "imgs_per_s"],
            [[t["batch"], t["imgs_per_s"]] for t in results["throughput"]],
        )
        + f"\n\nLatency (batch=1): P50 {results['latency']['p50_ms']} ms | "
        f"P95 {results['latency']['p95_ms']} ms\n",
        encoding="utf-8",
    )
    log.info("Benchmark -> %s", out_dir)


if __name__ == "__main__":
    main()
