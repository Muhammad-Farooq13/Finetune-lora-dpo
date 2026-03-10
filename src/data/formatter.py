"""Prompt formatting — convert parsed conversations into model-ready strings.

Uses the tokenizer's built-in chat template whenever possible so the output
is identical to what the model saw during pre-training. Falls back to manual
ChatML templating for unknown architectures.

The function schemas are embedded into the system prompt so the model has
them in-context during every training step.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


_SYSTEM_PREFIX = (
    "You are a helpful assistant that calls functions when appropriate.\n"
    "Available functions:\n\n"
)

_FUNCTION_CALL_INSTRUCTION = (
    "\nWhen calling a function respond ONLY with valid JSON in this exact format:\n"
    '{"name": "<function_name>", "arguments": {<key>: <value>, ...}}\n'
    "Do not include any explanation or extra text."
)


def build_system_prompt(functions: list[dict]) -> str:
    """Construct the system prompt with function schemas embedded."""
    if not functions:
        return "You are a helpful assistant."
    schemas = json.dumps(functions, indent=2)
    return _SYSTEM_PREFIX + schemas + _FUNCTION_CALL_INSTRUCTION


def format_for_sft(
    functions: list[dict],
    turns: list[dict[str, str]],
    tokenizer: "PreTrainedTokenizer",
) -> str:
    """Format a parsed conversation into a single training string.

    Applies the tokenizer's chat template so we stay consistent with the
    model's pre-training format. Tool response turns are mapped to the
    "user" role because most model templates don't have a native "tool" role.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(functions)}
    ]
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "tool":
            role = "user"
            content = f"[FUNCTION RESPONSE]: {content}"
        if role in ("user", "assistant", "system"):
            messages.append({"role": role, "content": content})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return _chatml_fallback(messages)


def _chatml_fallback(messages: list[dict[str, str]]) -> str:
    """Manual ChatML template for tokenizers without apply_chat_template."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("")  # trailing newline
    return "\n".join(parts)


def extract_function_call_from_output(text: str) -> str | None:
    """Pull out a JSON function call from assistant generated text."""
    # First try: look for the explicit <functioncall> tag
    match = re.search(r"<functioncall>\s*(\{.*?\})", text, re.DOTALL)
    if match:
        return match.group(1)
    # Second try: look for a JSON object with "name" and "arguments"
    match = re.search(
        r'(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\s*\})', text, re.DOTALL
    )
    if match:
        return match.group(1)
    return None
