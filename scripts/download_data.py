"""Download and preprocess glaive-function-calling-v2 from HuggingFace Hub.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --max-samples 5000   # quick dev run
    python scripts/download_data.py --output-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.config.training_config import SFTTrainingConfig
from src.data.loader import load_glaive_dataset

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download glaive-function-calling-v2")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional HuggingFace repo ID to push the processed dataset (e.g. user/my-dataset)",
    )
    args = parser.parse_args()

    config = SFTTrainingConfig()
    dataset = load_glaive_dataset(config, max_samples=args.max_samples)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))

    stats = {
        "train": len(dataset["train"]),
        "validation": len(dataset["validation"]),
        "features": list(dataset["train"].features.keys()),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nSaved to: {out.resolve()}")
    print(f"  Train samples:      {stats['train']:,}")
    print(f"  Validation samples: {stats['validation']:,}")

    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=True)
        print(f"\nPushed to Hub: https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
