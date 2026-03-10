"""Tests for DPO preference pair construction."""
from __future__ import annotations

import json

import pytest
from src.data.preference_builder import (
    _parse_function_call,
    _reject_hallucinate,
    _reject_missing_params,
    _reject_wrong_function,
    _reject_wrong_params,
    build_dpo_pairs,
    build_single_dpo_pair,
)
import random


_CALL = {"name": "get_weather", "arguments": {"location": "Lahore", "units": "celsius"}}
_AVAILABLE = ["get_weather", "search_products", "create_calendar_event"]


class TestParseFunctionCall:
    def test_plain_json(self):
        text = json.dumps(_CALL)
        result = _parse_function_call(text)
        assert result == _CALL

    def test_functioncall_tag(self):
        text = f"<functioncall> {json.dumps(_CALL)}"
        result = _parse_function_call(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_returns_none_for_plain_text(self):
        assert _parse_function_call("It is sunny today.") is None


class TestRejectionStrategies:
    def test_wrong_function_changes_name(self):
        rng = random.Random(0)
        result = _reject_wrong_function(_CALL, _AVAILABLE, rng)
        assert result is not None
        assert result["name"] != "get_weather"
        assert result["name"] in _AVAILABLE

    def test_wrong_function_no_alternatives(self):
        rng = random.Random(0)
        assert _reject_wrong_function(_CALL, ["get_weather"], rng) is None

    def test_wrong_params_same_name(self):
        rng = random.Random(0)
        result = _reject_wrong_params(_CALL, rng)
        assert result is not None
        assert result["name"] == "get_weather"
        # Values should differ from original
        assert result["arguments"]["location"] != "Lahore"

    def test_wrong_params_empty_args(self):
        rng = random.Random(0)
        call_no_args = {"name": "fn", "arguments": {}}
        assert _reject_wrong_params(call_no_args, rng) is None

    def test_missing_params_reduces_args(self):
        rng = random.Random(0)
        result = _reject_missing_params(_CALL, rng)
        assert result is not None
        assert len(result["arguments"]) < len(_CALL["arguments"])

    def test_missing_params_single_arg(self):
        rng = random.Random(0)
        call_one = {"name": "fn", "arguments": {"x": "val"}}
        assert _reject_missing_params(call_one, rng) is None

    def test_hallucinate_different_name(self):
        rng = random.Random(0)
        result = _reject_hallucinate(_CALL, rng)
        assert result is not None
        assert result["name"] != "get_weather"


class TestBuildSingleDpoPair:
    def test_returns_triplet_keys(self):
        chosen = json.dumps(_CALL)
        pair = build_single_dpo_pair("What is the weather?", chosen, _AVAILABLE, seed=42)
        assert pair is not None
        assert {"prompt", "chosen", "rejected", "rejection_strategy"}.issubset(pair.keys())

    def test_chosen_and_rejected_differ(self):
        chosen = json.dumps(_CALL)
        pair = build_single_dpo_pair("Query", chosen, _AVAILABLE, seed=0)
        assert pair is not None
        assert pair["chosen"] != pair["rejected"]

    def test_invalid_chosen_returns_none(self):
        pair = build_single_dpo_pair("Query", "not json at all", _AVAILABLE)
        assert pair is None

    def test_specific_strategy(self):
        chosen = json.dumps(_CALL)
        pair = build_single_dpo_pair(
            "Query", chosen, _AVAILABLE, strategy="wrong_function", seed=1
        )
        assert pair is not None
        assert pair["rejection_strategy"] == "wrong_function"
        rejected = json.loads(pair["rejected"])
        assert rejected["name"] != "get_weather"


class TestBuildDpoPairs:
    def _make_examples(self, n: int) -> list[dict]:
        return [
            {
                "prompt": f"Request {i}",
                "completion": json.dumps(_CALL),
                "available_functions": _AVAILABLE,
            }
            for i in range(n)
        ]

    def test_produces_pairs(self):
        examples = self._make_examples(20)
        pairs = build_dpo_pairs(examples, seed=0)
        assert len(pairs) > 0

    def test_all_pairs_have_required_keys(self):
        examples = self._make_examples(10)
        pairs = build_dpo_pairs(examples, seed=0)
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair

    def test_no_identical_chosen_rejected(self):
        examples = self._make_examples(20)
        pairs = build_dpo_pairs(examples, seed=0)
        for pair in pairs:
            assert pair["chosen"] != pair["rejected"]
