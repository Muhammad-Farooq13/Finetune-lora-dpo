"""Build and apply LoRA adapters using the PEFT library."""
from __future__ import annotations

import structlog
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

from src.config.training_config import LoRAConfig

logger = structlog.get_logger(__name__)

# Architecture → recommended target modules
# Targeting all projection layers (not just q/v) gives better results at the
# cost of slightly more trainable parameters — worth it for function calling.
_TARGET_MODULES: dict[str, list[str]] = {
    "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "v_proj"],
}


def detect_target_modules(model: PreTrainedModel) -> list[str]:
    """Infer the correct LoRA target modules from the model class name."""
    arch = type(model).__name__.lower()
    for key in _TARGET_MODULES:
        if key in arch:
            return _TARGET_MODULES[key]
    logger.warning("unknown_architecture", arch=arch, fallback="default")
    return _TARGET_MODULES["default"]


def build_peft_config(lora_cfg: LoRAConfig, model: PreTrainedModel) -> LoraConfig:
    """Create a PEFT LoraConfig, auto-detecting target modules if not specified."""
    target_modules = lora_cfg.target_modules or detect_target_modules(model)
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg.use_rslora,
    )


def apply_lora(
    model: PreTrainedModel,
    lora_cfg: LoRAConfig,
    is_quantized: bool = False,
) -> PreTrainedModel:
    """Wrap a model with LoRA adapters.

    For QLoRA (quantised) models we call prepare_model_for_kbit_training first
    to set up gradient checkpointing correctly with bitsandbytes.
    """
    if is_quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    peft_config = build_peft_config(lora_cfg, model)
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logger.info(
        "lora_applied",
        r=lora_cfg.r,
        alpha=lora_cfg.lora_alpha,
        target_modules=peft_config.target_modules,
        trainable_params=f"{trainable / 1e6:.1f}M",
        total_params=f"{total / 1e9:.2f}B",
        trainable_pct=f"{100 * trainable / total:.2f}%",
    )
    model.print_trainable_parameters()
    return model
