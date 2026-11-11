"""Start the A/B inference API locally."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--models", default=None,
        help='Spec like "lora=outputs/exp03_lora_r16_vit,linear=outputs/exp01_linear_probe_vit"',
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.models:
        os.environ["LVP_MODELS"] = args.models
    os.environ.setdefault("LVP_DEVICE", args.device)

    import uvicorn

    uvicorn.run("vision_pipeline.serving.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
