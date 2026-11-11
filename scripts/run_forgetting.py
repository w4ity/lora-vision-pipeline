"""Catastrophic-forgetting probe: feature drift + kNN accuracy on CIFAR-10."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.evaluation.forgetting import run_forgetting
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-train", type=int, default=2000,
                        help="CIFAR-10 train subset size for the kNN reference")
    parser.add_argument("--n-test", type=int, default=1000,
                        help="CIFAR-10 test subset size")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = run_forgetting(
        args.run_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        k=args.k,
        device=args.device,
    )
    log.info("Report -> %s", Path(out_dir) / "report.md")


if __name__ == "__main__":
    main()
