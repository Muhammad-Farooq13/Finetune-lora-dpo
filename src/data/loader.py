"""Dataset loading and preprocessing for function-calling fine-tuning.

Source dataset: glaiveai/glaive-function-calling-v2
  HuggingFace: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

Dataset format (raw):
  - system: "SYSTEM: You are a helpful assistant with access to the following
              functions. Use them if required -\\n\\n{...json schema...}"
  - chat:   "USER: <question>\\nASSISTANT: <functioncall> {...} </s>\\n
              FUNCTION RESPONSE: {...}\\nASSISTANT: <natural language reply> </s>"

This module parses both fields and produces structured examples containing:
  - functions:      JSON string of list[dict]  (function schemas)
  - turns:          JSON string of list[dict]  (conversation turns)
  - function_calls: JSON string of list[dict]  (ground-truth function calls)
  - has_function_call: bool
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog
from datasets import DatasetDict, load_dataset

from src.config.training_config import SFTTrainingConfig

logger = structlog.get_logger(__name__)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_system_functions(system_text: str) -> list[dict]:
    """Extract JSON function schemas from the system prompt string."""
    # The system field contains one or more JSON objects describing functions.
    # We pull out all top-level JSON objects and keep those with "name" + "description".
    functions: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(system_text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = system_text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "name" in obj and "description" in obj:
                        functions.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return functions


def _parse_conversation(chat_text: str) -> list[dict[str, str]]:
    """Parse the glaive chat format into a list of role/content dicts."""
    turns: list[dict[str, str]] = []
    # Split on role markers while keeping the marker
    parts = re.split(r"(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_text.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            content = part[5:].strip().removesuffix("</s>").strip()
            turns.append({"role": "user", "content": content})
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip().removesuffix("</s>").strip()
            turns.append({"role": "assistant", "content": content})
        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[18:].strip().removesuffix("</s>").strip()
            turns.append({"role": "tool", "content": content})
    return turns


def _extract_function_calls(turns: list[dict[str, str]]) -> list[dict]:
    """Pull out all <functioncall> JSON objects from assistant turns."""
    calls: list[dict] = []
    for turn in turns:
        if turn["role"] != "assistant":
            continue
        match = re.search(r"<functioncall>\s*(\{.*?\})", turn["content"], re.DOTALL)
        if match:
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
    return calls


# ── Public API ─────────────────────────────────────────────────────────────────

def load_glaive_dataset(
    config: SFTTrainingConfig,
    max_samples: int | None = None,
) -> DatasetDict:
    """Download and preprocess glaive-function-calling-v2.

    Returns a DatasetDict with "train" and "validation" splits.
    Each example has: functions, turns, function_calls, num_turns, has_function_call.
    """
    logger.info("loading_dataset", name=config.dataset_name)

    raw = load_dataset(config.dataset_name, split=config.dataset_split, trust_remote_code=True)

    cap = max_samples or config.max_train_samples
    if cap is not None:
        raw = raw.select(range(min(cap, len(raw))))

    def _preprocess(example: dict[str, Any]) -> dict[str, Any]:
        functions = _parse_system_functions(example.get("system", ""))
        turns = _parse_conversation(example.get("chat", ""))
        function_calls = _extract_function_calls(turns)
        return {
            "functions": json.dumps(functions),
            "turns": json.dumps(turns),
            "function_calls": json.dumps(function_calls),
            "num_turns": len(turns),
            "has_function_call": len(function_calls) > 0,
        }

    processed = raw.map(
        _preprocess,
        num_proc=4,
        desc="Parsing glaive conversations",
        remove_columns=raw.column_names,
    )

    # Keep only examples that have at least one function call — these are the
    # training signal; pure conversation examples don't help for our task.
    processed = processed.filter(lambda x: x["has_function_call"])

    split = processed.train_test_split(test_size=config.val_split_ratio, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info(
        "dataset_ready",
        train=len(dataset["train"]),
        validation=len(dataset["validation"]),
    )
    return dataset
