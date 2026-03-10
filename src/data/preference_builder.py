"""Build (prompt, chosen, rejected) triplets for DPO training.

glaive-function-calling-v2 only contains correct completions, so we
generate rejected completions synthetically using four strategies:

  wrong_function  — swap function name for a different one from the schema
  wrong_params    — corrupt argument values while keeping the function name
  missing_params  — drop half the required parameters
  hallucinate     — call a function name not present in the schema

Each strategy produces a plausible-looking-but-wrong call, which is what
DPO needs: the rejected completion should be in-distribution (not garbage)
so the model has to learn a real preference, not just "don't output noise".
"""
from __future__ import annotations

import json
import random
import re
from typing import Literal

RejectionStrategy = Literal[
    "wrong_function", "wrong_params", "missing_params", "hallucinate"
]

# Generic function names used for hallucination strategy
_HALLUCINATED_NAMES = [
    "send_email",
    "play_music",
    "control_device",
    "access_database",
    "delete_file",
    "call_contact",
    "print_document",
    "take_screenshot",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_dpo_pairs(
    examples: list[dict],
    seed: int = 42,
) -> list[dict[str, str]]:
    """Convert a list of SFT examples into DPO (prompt, chosen, rejected) triplets.

    Each input example must have keys: prompt, completion, available_functions.
    Returns a flat list of DPO triplets (some examples may produce 0 or >1 pairs).
    """
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []

    for example in examples:
        prompt = example.get("prompt", "")
        chosen = example.get("completion", "")
        available = example.get("available_functions", [])

        call = _parse_function_call(chosen)
        if call is None:
            continue

        # Attempt each strategy in random order; take the first successful one
        strategies: list[RejectionStrategy] = list(
            rng.sample(
                ["wrong_function", "wrong_params", "missing_params", "hallucinate"], 4
            )
        )
        for strategy in strategies:
            rejected_call = _apply_strategy(call, available, strategy, rng)
            if rejected_call is not None:
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": json.dumps(rejected_call),
                        "rejection_strategy": strategy,
                    }
                )
                break  # one rejected per chosen is enough

    return pairs


def build_single_dpo_pair(
    prompt: str,
    chosen_completion: str,
    available_function_names: list[str],
    strategy: RejectionStrategy | None = None,
    seed: int | None = None,
) -> dict[str, str] | None:
    """Convenience function for building a single DPO pair."""
    rng = random.Random(seed)
    call = _parse_function_call(chosen_completion)
    if call is None:
        return None

    if strategy is None:
        strategy = rng.choice(
            ["wrong_function", "wrong_params", "missing_params", "hallucinate"]
        )

    rejected_call = _apply_strategy(call, available_function_names, strategy, rng)
    if rejected_call is None:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_completion,
        "rejected": json.dumps(rejected_call),
        "rejection_strategy": strategy,
    }


# ── Strategy implementations ───────────────────────────────────────────────────

def _apply_strategy(
    call: dict,
    available: list[str],
    strategy: RejectionStrategy,
    rng: random.Random,
) -> dict | None:
    if strategy == "wrong_function":
        return _reject_wrong_function(call, available, rng)
    if strategy == "wrong_params":
        return _reject_wrong_params(call, rng)
    if strategy == "missing_params":
        return _reject_missing_params(call, rng)
    if strategy == "hallucinate":
        return _reject_hallucinate(call, rng)
    return None


def _reject_wrong_function(call: dict, available: list[str], rng: random.Random) -> dict | None:
    """Substitute a different function name from the given schema."""
    alternatives = [f for f in available if f != call.get("name")]
    if not alternatives:
        return None
    return {"name": rng.choice(alternatives), "arguments": call.get("arguments", {})}


def _reject_wrong_params(call: dict, rng: random.Random) -> dict | None:
    """Corrupt argument values while keeping function name and keys."""
    args = call.get("arguments", {})
    if not args:
        return None

    corrupted: dict = {}
    for key, val in args.items():
        if isinstance(val, str) and len(val) > 3:
            # Reverse the string — looks plausible but is wrong
            corrupted[key] = val[::-1]
        elif isinstance(val, (int, float)):
            corrupted[key] = -(abs(val) + 42)
        elif isinstance(val, bool):
            corrupted[key] = not val
        elif isinstance(val, list) and val:
            corrupted[key] = list(reversed(val))
        else:
            corrupted[key] = "invalid_value"

    return {"name": call["name"], "arguments": corrupted}


def _reject_missing_params(call: dict, rng: random.Random) -> dict | None:
    """Drop half the parameters — simulates forgetting required fields."""
    args = call.get("arguments", {})
    if len(args) < 2:
        return None
    keys = list(args.keys())
    rng.shuffle(keys)
    keep = keys[: max(1, len(keys) // 2)]
    return {"name": call["name"], "arguments": {k: args[k] for k in keep}}


def _reject_hallucinate(call: dict, rng: random.Random) -> dict:
    """Call a function that does not exist in the schema."""
    fake_name = rng.choice(_HALLUCINATED_NAMES)
    return {
        "name": fake_name,
        "arguments": {"query": "example", "target": "user"},
    }


# ── Parse helper ──────────────────────────────────────────────────────────────

def _parse_function_call(text: str) -> dict | None:
    """Extract the first JSON function call from a completion string."""
    for pattern in [
        r"<functioncall>\s*(\{.*?\})",
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
