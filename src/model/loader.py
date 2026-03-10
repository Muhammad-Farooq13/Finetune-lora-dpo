"""Model and tokenizer loading with optional QLoRA quantisation."""
from __future__ import annotations

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config.training_config import QuantizationConfig, SFTTrainingConfig

logger = structlog.get_logger(__name__)


def load_tokenizer(model_name: str, padding_side: str = "right") -> PreTrainedTokenizer:
    """Load tokenizer and ensure pad_token is set."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side=padding_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_name: str,
    quant_config: QuantizationConfig | None = None,
    device_map: str | dict = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> PreTrainedModel:
    """Load the base model, optionally in 4-bit NF4 for QLoRA.

    For QLoRA we pass a BitsAndBytesConfig; for full precision we pass
    torch_dtype directly. The two are mutually exclusive in transformers.
    """
    bnb_config: BitsAndBytesConfig | None = None
    if quant_config is not None and quant_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        )
        logger.info("qlora_quantisation_enabled", quant_type=quant_config.bnb_4bit_quant_type)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype if bnb_config is None else None,
        trust_remote_code=True,
        # Use "flash_attention_2" if flash-attn is installed and you're on Linux+CUDA
        attn_implementation="eager",
    )

    if bnb_config is not None:
        # Required for gradient checkpointing to work with quantised models
        model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(
        "base_model_loaded",
        name=model_name,
        total_params_b=f"{total_params:.2f}",
        dtype=str(model.dtype),
        quantised=bnb_config is not None,
    )
    return model


def load_model_and_tokenizer(
    config: SFTTrainingConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Convenience wrapper that loads both model and tokenizer from a config."""
    tokenizer = load_tokenizer(config.model_name)
    quant = QuantizationConfig() if config.load_in_4bit else None
    model = load_base_model(config.model_name, quant_config=quant)
    return model, tokenizer
