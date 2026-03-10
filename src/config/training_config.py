"""Training configuration dataclasses for LoRA, SFT, and DPO."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    r=16, alpha=32 is a reliable default for most 1–7B models on instruction
    following tasks. Increase r for tasks that need more capacity (code, math).
    """

    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False  # Rank-Stabilised LoRA — useful when r > 64


@dataclass
class QuantizationConfig:
    """BitsAndBytes NF4 quantisation for QLoRA.

    NF4 + double quantisation gives roughly the same perplexity as BF16
    at ~4× lower VRAM, which is the whole point of QLoRA.
    """

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True  # nested quantisation
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"


@dataclass
class SFTTrainingConfig:
    """Supervised fine-tuning hyperparameters."""

    # Model & data
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    dataset_name: str = "glaiveai/glaive-function-calling-v2"
    dataset_split: str = "train"
    val_split_ratio: float = 0.05
    max_train_samples: int | None = None  # None = full dataset

    # Output
    output_dir: str = field(
        default_factory=lambda: os.getenv("SFT_OUTPUT_DIR", "./checkpoints/sft")
    )
    run_name: str = "sft-lora-function-calling"

    # Training loop
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    # Optimiser
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0

    # LR schedule
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Sequence
    max_seq_length: int = 2048

    # Logging & checkpointing
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Precision
    bf16: bool = True
    tf32: bool = True

    # Sequence packing — SFTTrainer bins short sequences together for throughput
    packing: bool = True

    # QLoRA
    load_in_4bit: bool = field(
        default_factory=lambda: os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DPOTrainingConfig:
    """Direct Preference Optimisation hyperparameters.

    DPO is sensitive to beta: too low → policy collapses to reward hacking;
    too high → essentially no update from SFT baseline.
    0.1 is a safe default for most instruction-following tasks.
    """

    # Model paths
    sft_model_path: str = "./checkpoints/sft"
    output_dir: str = field(
        default_factory=lambda: os.getenv("DPO_OUTPUT_DIR", "./checkpoints/dpo")
    )
    run_name: str = "dpo-function-calling"

    # DPO-specific
    beta: float = 0.1
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"
    label_smoothing: float = 0.0  # > 0 for conservative DPO

    # Training loop
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True

    # Optimiser
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LR schedule
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"

    # Sequence lengths
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_target_length: int = 512

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2

    # Precision
    bf16: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
