"""Run SFT with LoRA/QLoRA.

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --lr 2e-4
    python scripts/train_sft.py --data-dir data/processed --max-samples 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.config.training_config import LoRAConfig, SFTTrainingConfig
from src.data.loader import load_glaive_dataset
from src.model.loader import load_model_and_tokenizer
from src.training.sft_trainer import prepare_sft_dataset, run_sft

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training with LoRA/QLoRA")
    p.add_argument("--model", default=None, help="Override MODEL_NAME env var")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--data-dir", default=None, help="Load pre-saved dataset from disk")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-4bit", action="store_true", help="Disable QLoRA (requires more VRAM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build configs (CLI args override env/dataclass defaults)
    sft_config = SFTTrainingConfig()
    if args.model:
        sft_config.model_name = args.model
    if args.epochs:
        sft_config.num_train_epochs = args.epochs
    if args.lr:
        sft_config.learning_rate = args.lr
    if args.max_samples:
        sft_config.max_train_samples = args.max_samples
    if args.output_dir:
        sft_config.output_dir = args.output_dir
    if args.no_4bit:
        sft_config.load_in_4bit = False

    lora_config = LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha)

    logger.info(
        "sft_config",
        model=sft_config.model_name,
        epochs=sft_config.num_train_epochs,
        lr=sft_config.learning_rate,
        lora_r=lora_config.r,
        qlora=sft_config.load_in_4bit,
    )

    # Load data
    if args.data_dir:
        from datasets import load_from_disk
        dataset = load_from_disk(args.data_dir)
        logger.info("loaded_from_disk", path=args.data_dir)
    else:
        dataset = load_glaive_dataset(sft_config)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(sft_config)

    # Format dataset
    dataset = prepare_sft_dataset(dataset, tokenizer)

    # Train
    _, metrics = run_sft(model, tokenizer, dataset, sft_config, lora_config)

    print(f"\nSFT complete. Checkpoint saved to: {sft_config.output_dir}")
    print(f"  Final train loss: {metrics.get('train_loss', 'n/a'):.4f}")


if __name__ == "__main__":
    main()
