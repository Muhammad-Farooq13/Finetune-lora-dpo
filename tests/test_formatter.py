"""Tests for data formatting."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from src.data.formatter import (
    _chatml_fallback,
    build_system_prompt,
    extract_function_call_from_output,
    format_for_sft,
)

_SAMPLE_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
]

_SAMPLE_TURNS = [
    {"role": "user", "content": "What is the weather in Lahore?"},
    {"role": "assistant", "content": '{"name": "get_weather", "arguments": {"location": "Lahore"}}'},
]


class TestBuildSystemPrompt:
    def test_includes_function_name(self):
        prompt = build_system_prompt(_SAMPLE_FUNCTIONS)
        assert "get_weather" in prompt

    def test_no_functions(self):
        prompt = build_system_prompt([])
        assert prompt == "You are a helpful assistant."

    def test_includes_json_instruction(self):
        prompt = build_system_prompt(_SAMPLE_FUNCTIONS)
        assert "JSON" in prompt


class TestFormatForSft:
    def test_uses_tokenizer_template(self):
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "<formatted>"
        result = format_for_sft(_SAMPLE_FUNCTIONS, _SAMPLE_TURNS, mock_tok)
        assert result == "<formatted>"
        mock_tok.apply_chat_template.assert_called_once()

    def test_fallback_on_exception(self):
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = Exception("unsupported")
        result = format_for_sft(_SAMPLE_FUNCTIONS, _SAMPLE_TURNS, mock_tok)
        # Should fall back to ChatML with im_start tags
        assert "<|im_start|>" in result

    def test_tool_role_mapped_to_user(self):
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.side_effect = Exception("unsupported")
        turns_with_tool = [
            {"role": "user", "content": "What time is it?"},
            {"role": "tool", "content": '{"time": "12:00"}'},
            {"role": "assistant", "content": "It is noon."},
        ]
        result = format_for_sft([], turns_with_tool, mock_tok)
        # tool role should appear as user in the fallback
        assert "FUNCTION RESPONSE" in result or "<|im_start|>user" in result


class TestExtractFunctionCall:
    def test_plain_json(self):
        text = '{"name": "get_weather", "arguments": {"location": "Lahore"}}'
        result = extract_function_call_from_output(text)
        assert result is not None
        assert "get_weather" in result

    def test_functioncall_tag(self):
        text = "Sure! <functioncall> {\"name\": \"fn\", \"arguments\": {}} </s>"
        result = extract_function_call_from_output(text)
        assert result is not None

    def test_no_call(self):
        result = extract_function_call_from_output("The weather is sunny.")
        assert result is None


class TestChatmlFallback:
    def test_has_im_start_tags(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _chatml_fallback(messages)
        assert result.count("<|im_start|>") == 3
        assert result.count("<|im_end|>") == 3
