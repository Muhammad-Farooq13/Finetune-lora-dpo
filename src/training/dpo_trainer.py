"""DPO training using TRL's DPOTrainer.

Workflow:
  1. Load the SFT checkpoint (a LoRA adapter on top of the base model).
  2. Merge the SFT LoRA into the base model weights.
  3. Apply a fresh LoRA for DPO fine-tuning.
  4. Train with DPOTrainer — pass ref_model=None so TRL handles the reference
     internally by disabling/enabling adapters (memory-efficient approach).
  5. Save the DPO adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

from src.config.training_config import DPOTrainingConfig, LoRAConfig, QuantizationConfig
from src.model.loader import load_tokenizer
from src.model.lora_config import apply_lora

logger = structlog.get_logger(__name__)


def load_and_merge_sft(sft_path: str) -> tuple:
    """Load the SFT PEFT model, merge LoRA weights into the base, return merged model."""
    logger.info("loading_sft_checkpoint", path=sft_path)
    tokenizer = load_tokenizer(sft_path)

    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # merge_and_unload folds the LoRA delta into the base weights and returns a
    # plain PreTrainedModel — required before applying a second LoRA for DPO.
    logger.info("merging_lora_weights")
    merged = peft_model.merge_and_unload()
    return merged, tokenizer


def run_dpo(
    dataset: Dataset,
    config: DPOTrainingConfig,
    lora_config: LoRAConfig,
) -> tuple:
    """Run DPO on top of the merged SFT model.

    dataset must have columns: prompt, chosen, rejected.
    """
    merged_model, tokenizer = load_and_merge_sft(config.sft_model_path)

    # Apply a fresh LoRA for DPO so the reference model is the merged SFT model
    # and we only update a small adapter during DPO
    dpo_model = apply_lora(merged_model, lora_config, is_quantized=False)

    dpo_args = DPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
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
        bf16=config.bf16,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        eval_strategy="no",
        report_to=["wandb"] if _wandb_available() else ["none"],
    )

    # ref_model=None → TRL will disable the LoRA adapter on the policy model to
    # obtain reference log-probs (efficient; avoids loading a second full model).
    trainer = DPOTrainer(
        model=dpo_model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info(
        "dpo_training_start",
        beta=config.beta,
        loss_type=config.loss_type,
        samples=len(dataset),
    )
    train_result = trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    Path(config.output_dir, "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("dpo_training_complete", **{k: round(v, 4) for k, v in metrics.items()})
    return dpo_model, metrics


def _wandb_available() -> bool:
    try:
        import os
        import wandb  # noqa: F401
        return bool(os.getenv("WANDB_API_KEY"))
    except ImportError:
        return False
