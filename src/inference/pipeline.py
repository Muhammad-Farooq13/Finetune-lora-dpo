"""Inference pipeline — load a trained LoRA adapter and run predictions."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import structlog
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

from src.data.formatter import build_system_prompt

logger = structlog.get_logger(__name__)


@dataclass
class FunctionCallResult:
    """Result of a single inference call."""

    raw_output: str
    function_name: str | None
    arguments: dict
    latency_ms: float
    valid: bool
    error: str | None = None


class FunctionCallingPipeline:
    """Load a fine-tuned LoRA adapter and produce function call predictions.

    Usage:
        pipeline = FunctionCallingPipeline("checkpoints/dpo")
        result = pipeline.call(
            user_message="What is the weather in Lahore?",
            available_functions=[{"name": "get_weather", ...}],
        )
        print(result.function_name, result.arguments)
    """

    def __init__(
        self,
        adapter_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        logger.info("loading_pipeline", adapter=adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        extra: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **extra,
        )
        self.model.eval()
        logger.info("pipeline_ready")

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(
        self,
        user_message: str,
        available_functions: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> FunctionCallResult:
        """Generate a function call for the given user request."""
        prompt = self._build_prompt(user_message, available_functions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
        parsed = self._parse(raw)

        logger.info(
            "inference_done",
            latency_ms=round(latency_ms, 1),
            valid=parsed is not None,
            fn=parsed.get("name") if parsed else None,
        )

        return FunctionCallResult(
            raw_output=raw,
            function_name=parsed.get("name") if parsed else None,
            arguments=parsed.get("arguments", {}) if parsed else {},
            latency_ms=round(latency_ms, 1),
            valid=parsed is not None,
        )

    def stream(
        self,
        user_message: str,
        available_functions: list[dict],
        max_new_tokens: int = 256,
    ) -> Iterator[str]:
        """Yield tokens as they are generated (for interactive use)."""
        prompt = self._build_prompt(user_message, available_functions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        yield from streamer
        thread.join()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_prompt(self, user_message: str, functions: list[dict]) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt(functions)},
            {"role": "user", "content": user_message},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to simple ChatML
            return (
                f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _parse(self, text: str) -> dict | None:
        """Extract function call JSON from generated text."""
        for pattern in [
            r'(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\s*\})',
            r"(\{[^{}]*\"name\"[^{}]*\})",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None
