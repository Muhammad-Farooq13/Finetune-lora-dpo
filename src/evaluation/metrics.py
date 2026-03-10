"""Evaluation metrics for function-calling and JSON extraction quality.

Metrics computed:
  valid_json_rate       % of outputs that parse as valid JSON
  exact_tool_match_rate % where predicted function name == ground truth name
  parameter_f1          token-level F1 between predicted and reference argument values
  hallucination_rate    % where predicted function is not in the allowed schema
  rouge_l               sentence-level ROUGE-L F1 on the full completion
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionCallMetrics:
    """Accumulator for batch evaluation metrics."""

    n: int = 0
    valid_json: int = 0
    exact_tool_match: int = 0
    hallucinated: int = 0
    _param_precision_sum: float = field(default=0.0, repr=False)
    _param_recall_sum: float = field(default=0.0, repr=False)
    _rouge_l_sum: float = field(default=0.0, repr=False)

    # ── Derived ────────────────────────────────────────────────────────────────

    @property
    def valid_json_rate(self) -> float:
        return 100.0 * self.valid_json / self.n if self.n else 0.0

    @property
    def exact_tool_match_rate(self) -> float:
        return 100.0 * self.exact_tool_match / self.n if self.n else 0.0

    @property
    def hallucination_rate(self) -> float:
        return 100.0 * self.hallucinated / self.n if self.n else 0.0

    @property
    def parameter_precision(self) -> float:
        return self._param_precision_sum / self.n if self.n else 0.0

    @property
    def parameter_recall(self) -> float:
        return self._param_recall_sum / self.n if self.n else 0.0

    @property
    def parameter_f1(self) -> float:
        p, r = self.parameter_precision, self.parameter_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def rouge_l(self) -> float:
        return self._rouge_l_sum / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "valid_json_rate": round(self.valid_json_rate, 2),
            "exact_tool_match_rate": round(self.exact_tool_match_rate, 2),
            "parameter_f1": round(self.parameter_f1, 4),
            "parameter_precision": round(self.parameter_precision, 4),
            "parameter_recall": round(self.parameter_recall, 4),
            "hallucination_rate": round(self.hallucination_rate, 2),
            "rouge_l": round(self.rouge_l, 4),
        }


# ── Core metric functions ──────────────────────────────────────────────────────

def parse_function_call(text: str) -> dict | None:
    """Try to extract a function call JSON object from model output."""
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

    # Last resort: try to parse the whole text as JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def compute_parameter_f1(pred_args: dict, ref_args: dict) -> tuple[float, float]:
    """Token-level precision and recall across all argument values.

    We flatten all values to token lists and compute micro-averaged P/R.
    This handles cases where a string value is partially correct.
    """

    def _tokens(d: dict) -> list[str]:
        tokens: list[str] = []
        for v in d.values():
            tokens.extend(str(v).lower().split())
        return tokens

    pred_tok = _tokens(pred_args)
    ref_tok = _tokens(ref_args)

    if not pred_tok and not ref_tok:
        return 1.0, 1.0
    if not pred_tok or not ref_tok:
        return 0.0, 0.0

    pred_counter = Counter(pred_tok)
    ref_counter = Counter(ref_tok)
    common = sum((pred_counter & ref_counter).values())

    precision = common / len(pred_tok)
    recall = common / len(ref_tok)
    return precision, recall


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Sentence-level ROUGE-L F1 between two strings."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """LCS via two-row DP (O(n) space)."""
    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for ai in a:
        for j, bj in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ai == bj else max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ── Per-example accumulation ───────────────────────────────────────────────────

def evaluate_single(
    prediction: str,
    reference: str,
    allowed_functions: list[str],
    metrics: FunctionCallMetrics,
) -> None:
    """Accumulate metrics for one prediction/reference pair (mutates metrics)."""
    metrics.n += 1
    metrics._rouge_l_sum += compute_rouge_l(prediction, reference)

    pred_call = parse_function_call(prediction)
    ref_call = parse_function_call(reference)

    if pred_call is None:
        # Failed to produce valid JSON — no further credit
        return

    metrics.valid_json += 1

    pred_name = pred_call.get("name", "")

    if allowed_functions and pred_name not in allowed_functions:
        metrics.hallucinated += 1

    if ref_call is None:
        return

    ref_name = ref_call.get("name", "")
    if pred_name == ref_name:
        metrics.exact_tool_match += 1

    p, r = compute_parameter_f1(
        pred_call.get("arguments", {}),
        ref_call.get("arguments", {}),
    )
    metrics._param_precision_sum += p
    metrics._param_recall_sum += r
