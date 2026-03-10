"""Run DPO on top of an SFT checkpoint.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --sft-checkpoint checkpoints/sft --beta 0.1
    python scripts/train_dpo.py --loss-type ipo --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from datasets import Dataset

from src.config.training_config import DPOTrainingConfig, LoRAConfig, SFTTrainingConfig
from src.data.loader import load_glaive_dataset
from src.data.formatter import format_for_sft
from src.data.preference_builder import build_dpo_pairs
from src.model.loader import load_tokenizer
from src.training.dpo_trainer import run_dpo

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO training")
    p.add_argument("--sft-checkpoint", default="./checkpoints/sft")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--loss-type", default="sigmoid", choices=["sigmoid", "hinge", "ipo", "kto_pair"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", default="./checkpoints/dpo")
    return p.parse_args()


def build_preference_dataset(sft_data, tokenizer) -> Dataset:
    """Convert SFT examples into DPO (prompt, chosen, rejected) triplets."""
    import json as _json

    raw_examples = []
    for example in sft_data:
        functions = _json.loads(example["functions"])
        turns = _json.loads(example["turns"])
        function_calls = _json.loads(example["function_calls"])
        if not function_calls:
            continue

        prompt = format_for_sft(functions, turns[:-1], tokenizer) if len(turns) > 1 else ""
        chosen = _json.dumps(function_calls[0])
        available = [fn.get("name", "") for fn in functions]

        raw_examples.append(
            {"prompt": prompt, "completion": chosen, "available_functions": available}
        )

    pairs = build_dpo_pairs(raw_examples)
    logger.info("dpo_pairs_built", n=len(pairs))
    return Dataset.from_list(pairs)


def main() -> None:
    args = parse_args()

    dpo_config = DPOTrainingConfig(
        sft_model_path=args.sft_checkpoint,
        beta=args.beta,
        loss_type=args.loss_type,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    lora_config = LoRAConfig(r=args.lora_r, lora_alpha=args.lora_r * 2)

    logger.info(
        "dpo_config",
        sft=args.sft_checkpoint,
        beta=dpo_config.beta,
        loss_type=dpo_config.loss_type,
    )

    # Load the same training data to build preference pairs
    sft_config = SFTTrainingConfig()
    if args.max_samples:
        sft_config.max_train_samples = args.max_samples
    sft_dataset = load_glaive_dataset(sft_config)

    tokenizer = load_tokenizer(args.sft_checkpoint)
    dpo_dataset = build_preference_dataset(sft_dataset["train"], tokenizer)

    _, metrics = run_dpo(dpo_dataset, dpo_config, lora_config)

    print(f"\nDPO complete. Checkpoint saved to: {dpo_config.output_dir}")


if __name__ == "__main__":
    main()
