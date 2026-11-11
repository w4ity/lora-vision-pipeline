"""Evaluate a trained run: clean test split + stress benchmark (corruptions)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.evaluation.evaluate import evaluate_run
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    clean = evaluate_run(args.run_dir, split=args.split,
                         severities=args.severities, device=args.device)
    log.info("Final: acc %.4f | macro_f1 %.4f", clean["accuracy"], clean["macro_f1"])


if __name__ == "__main__":
    main()
