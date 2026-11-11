"""Head-to-head comparison: run A (baseline) vs run B (LoRA) on the same manifest."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.evaluation.compare import compare_runs
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="Baseline run dir (linear probe)")
    parser.add_argument("--run-b", required=True, help="Treatment run dir (LoRA)")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = compare_runs(args.run_a, args.run_b, split=args.split,
                           severities=args.severities, device=args.device)
    log.info("Report -> %s", Path(out_dir) / "report.md")


if __name__ == "__main__":
    main()
