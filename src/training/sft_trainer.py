"""Supervised Fine-Tuning using TRL's SFTTrainer."""
from __future__ import annotations

import json
from pathlib import Path

import structlog
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from src.config.training_config import LoRAConfig, SFTTrainingConfig
from src.data.formatter import format_for_sft
from src.model.lora_config import apply_lora

logger = structlog.get_logger(__name__)


def prepare_sft_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """Format each example into the model's chat template string.

    SFTTrainer expects a "text" column containing the full formatted conversation.
    """
    import json as _json

    def _to_text(example: dict) -> dict:
        functions = _json.loads(example["functions"])
        turns = _json.loads(example["turns"])
        return {"text": format_for_sft(functions, turns, tokenizer)}

    return dataset_dict.map(_to_text, num_proc=4, desc="Applying chat template")


def run_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    config: SFTTrainingConfig,
    lora_config: LoRAConfig,
) -> tuple[PreTrainedModel, dict]:
    """Run supervised fine-tuning with LoRA/QLoRA.

    Returns the trained PEFT model and a metrics dict.
    """
    # Apply LoRA before building the trainer so the adapter is registered
    is_quantized = config.load_in_4bit
    model = apply_lora(model, lora_config, is_quantized=is_quantized)

    sft_args = SFTConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        bf16=config.bf16,
        tf32=config.tf32,
        max_seq_length=config.max_seq_length,
        packing=config.packing,
        eval_strategy="steps",
        dataset_text_field="text",
        report_to=["wandb"] if _wandb_available() else ["none"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
    )

    logger.info(
        "sft_training_start",
        model=config.model_name,
        train_samples=len(dataset["train"]),
        effective_batch=config.per_device_train_batch_size * config.gradient_accumulation_steps,
    )
    train_result = trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    Path(config.output_dir, "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("sft_training_complete", **{k: round(v, 4) for k, v in metrics.items()})
    return model, metrics


def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        import os
        return bool(os.getenv("WANDB_API_KEY"))
    except ImportError:
        return False
