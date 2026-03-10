"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest
from src.evaluation.metrics import (
    FunctionCallMetrics,
    compute_parameter_f1,
    compute_rouge_l,
    evaluate_single,
    parse_function_call,
)


# ── parse_function_call ────────────────────────────────────────────────────────

class TestParseFunctionCall:
    def test_parses_standard_json(self):
        text = '{"name": "get_weather", "arguments": {"location": "Lahore"}}'
        result = parse_function_call(text)
        assert result == {"name": "get_weather", "arguments": {"location": "Lahore"}}

    def test_parses_functioncall_tag(self):
        text = '<functioncall> {"name": "search", "arguments": {"q": "test"}} </s>'
        result = parse_function_call(text)
        assert result is not None
        assert result["name"] == "search"

    def test_returns_none_for_plain_text(self):
        assert parse_function_call("The weather in Lahore is sunny.") is None

    def test_returns_none_for_empty(self):
        assert parse_function_call("") is None

    def test_embedded_in_prose(self):
        text = 'Sure! {"name": "convert_currency", "arguments": {"amount": 100}} Done.'
        result = parse_function_call(text)
        assert result is not None
        assert result["name"] == "convert_currency"


# ── compute_parameter_f1 ──────────────────────────────────────────────────────

class TestParameterF1:
    def test_exact_match(self):
        args = {"location": "Lahore", "units": "celsius"}
        p, r = compute_parameter_f1(args, args)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_empty_both(self):
        p, r = compute_parameter_f1({}, {})
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_pred_empty(self):
        p, r = compute_parameter_f1({}, {"location": "Lahore"})
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)

    def test_partial_match(self):
        pred = {"location": "Lahore Pakistan"}
        ref = {"location": "Lahore"}
        p, r = compute_parameter_f1(pred, ref)
        # ref token "lahore" is in pred → recall = 1.0; pred has extra token → precision < 1
        assert r == pytest.approx(1.0)
        assert p < 1.0

    def test_no_overlap(self):
        p, r = compute_parameter_f1({"city": "Karachi"}, {"city": "Lahore"})
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)


# ── compute_rouge_l ───────────────────────────────────────────────────────────

class TestRougeL:
    def test_identical(self):
        assert compute_rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_rouge_l("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_empty_strings(self):
        assert compute_rouge_l("", "anything") == pytest.approx(0.0)
        assert compute_rouge_l("anything", "") == pytest.approx(0.0)

    def test_partial(self):
        score = compute_rouge_l("the cat sat", "the cat")
        assert 0.0 < score < 1.0


# ── FunctionCallMetrics accumulator ──────────────────────────────────────────

class TestFunctionCallMetrics:
    def test_empty(self):
        m = FunctionCallMetrics()
        assert m.valid_json_rate == 0.0
        assert m.parameter_f1 == 0.0

    def test_perfect_run(self):
        m = FunctionCallMetrics()
        for _ in range(10):
            evaluate_single(
                '{"name": "get_weather", "arguments": {"location": "Lahore"}}',
                '{"name": "get_weather", "arguments": {"location": "Lahore"}}',
                ["get_weather"],
                m,
            )
        assert m.valid_json_rate == pytest.approx(100.0)
        assert m.exact_tool_match_rate == pytest.approx(100.0)
        assert m.hallucination_rate == pytest.approx(0.0)

    def test_all_hallucinations(self):
        m = FunctionCallMetrics()
        for _ in range(5):
            evaluate_single(
                '{"name": "fake_function", "arguments": {}}',
                '{"name": "get_weather", "arguments": {"location": "X"}}',
                ["get_weather"],
                m,
            )
        assert m.hallucination_rate == pytest.approx(100.0)

    def test_invalid_json(self):
        m = FunctionCallMetrics()
        evaluate_single("This is not JSON at all.", '{"name": "fn", "arguments": {}}', [], m)
        assert m.valid_json == 0
        assert m.n == 1

    def test_to_dict_keys(self):
        m = FunctionCallMetrics()
        d = m.to_dict()
        expected = {
            "n", "valid_json_rate", "exact_tool_match_rate",
            "parameter_f1", "parameter_precision", "parameter_recall",
            "hallucination_rate", "rouge_l",
        }
        assert expected.issubset(d.keys())
