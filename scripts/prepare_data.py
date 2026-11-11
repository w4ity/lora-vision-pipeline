"""Download + materialize the dataset and build stratified train/val/test manifests."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.config import load_config
from vision_pipeline.data.datasets import (
    Manifest,
    create_manifests,
    dataset_stats,
    manifest_dir_for,
)
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Split seed (default: config seed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed

    paths = create_manifests(cfg.data, seed=seed)
    stats = {}
    for split, path in paths.items():
        manifest = Manifest.load(path)
        stats[split] = {"n": len(manifest.items), "per_class": dataset_stats(manifest)}
        log.info("%s: %d items", split, len(manifest.items))

    out = manifest_dir_for(cfg.data) / "stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Stats -> %s", out)
    log.info("Manifests ready under %s", manifest_dir_for(cfg.data))


if __name__ == "__main__":
    main()
