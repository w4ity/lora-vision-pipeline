"""Train a single experiment from its config file."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.config import load_config
from vision_pipeline.training.run import train_config
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir, best = train_config(cfg)

    log.info("Best epoch %d | val_macro_f1 %.4f | val_accuracy %.4f",
             best["epoch"], best["val_macro_f1"], best["val_accuracy"])
    log.info("Next: python scripts/evaluate.py --run-dir %s", run_dir)


if __name__ == "__main__":
    main()
