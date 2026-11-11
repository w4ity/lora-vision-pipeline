"""Run a sweep of experiment configs sequentially and emit a comparison table."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vision_pipeline.training.sweep import run_sweep
from vision_pipeline.utils import get_logger

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Experiment names (resolved under configs/experiments/) or YAML paths",
    )
    parser.add_argument("--out-dir", default="outputs/sweep")
    args = parser.parse_args()

    paths = []
    for c in args.configs:
        p = Path(c)
        if p.suffix != ".yaml":
            p = Path("configs/experiments") / f"{c}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        paths.append(p)

    run_sweep(paths, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
