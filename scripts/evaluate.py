"""Evaluate base, SFT, and DPO checkpoints and print a comparison table.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct \\
        --sft checkpoints/sft --dpo checkpoints/dpo --max-samples 500
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
from src.data.formatter import format_for_sft
from src.evaluation.evaluator import compare_models
from src.model.loader import load_tokenizer

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base vs SFT vs DPO")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sft", default="./checkpoints/sft")
    p.add_argument("--dpo", default="./checkpoints/dpo")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--output", default="experiments/results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("loading_test_data")
    config = SFTTrainingConfig(max_train_samples=5000)  # small load for eval
    dataset = load_glaive_dataset(config)
    test_split = dataset["validation"]

    # Add formatted text column (needed by evaluator for prompts)
    tokenizer = load_tokenizer(args.base)

    def add_text(example):
        import json as _json
        functions = _json.loads(example["functions"])
        turns = _json.loads(example["turns"])
        # Provide everything except the final assistant turn as the prompt
        prompt_turns = [t for t in turns if t["role"] != "assistant"]
        return {"text": format_for_sft(functions, prompt_turns, tokenizer)}

    test_split = test_split.map(add_text, desc="Formatting test prompts")

    report = compare_models(
        base_model=args.base,
        sft_model=args.sft,
        dpo_model=args.dpo,
        tokenizer_path=args.base,
        test_dataset=test_split,
        output_path=args.output,
        max_samples=args.max_samples,
    )

    print(f"\nFull results written to: {args.output}")

    # Show improvement summary
    r = report["results"]
    base_tm = r.get("base", {}).get("exact_tool_match_rate", 0)
    dpo_tm = r.get("dpo", {}).get("exact_tool_match_rate", 0)
    print(f"\nTool match improvement (base → DPO): {base_tm:.1f}% → {dpo_tm:.1f}% ({dpo_tm - base_tm:+.1f} pp)")


if __name__ == "__main__":
    main()
