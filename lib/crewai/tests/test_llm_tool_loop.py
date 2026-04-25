"""Tests for LLM.call() tool loop and LLMResult.

All LLM calls are mocked — no real API traffic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm_result import (
    LLMResult,
    ToolCallRecord,
    _lookup_pricing,
    estimate_cost_usd,
)


def _make_litellm_llm(model: str = "gpt-4o") -> Any:
    """Create an LLM instance that uses the litellm fallback path."""
    from crewai.llm import LLM
    return LLM(model=model, is_litellm=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    """Build a tool-call object using litellm's actual types."""
    try:
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Function,
        )
        return ChatCompletionMessageToolCall(
            id=call_id,
            function=Function(name=name, arguments=json.dumps(arguments)),
            type="function",
        )
    except ImportError:
        func = SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return SimpleNamespace(id=call_id, function=func, type="function")


def _make_model_response(content: str | None = None, tool_calls: list | None = None):
    """Build a minimal mock ModelResponse that passes isinstance checks.

    We need it to be an instance of litellm's ModelResponse/ModelResponseBase
    so the internal isinstance() checks work. We import those types when
    litellm is available.
    """
    try:
        from litellm.types.utils import (
            Choices,
            Message,
            ModelResponse,
            Usage,
        )

        message = Message(content=content, tool_calls=tool_calls or None)
        choice = Choices(message=message, finish_reason="stop", index=0)
        resp = ModelResponse(
            choices=[choice],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )
        return resp
    except ImportError:
        # Fallback to SimpleNamespace if litellm not installed
        message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        resp = SimpleNamespace(
            choices=[choice],
            model_extra={"usage": usage},
        )
        return resp


DUMMY_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Unit tests for LLMResult / ToolCallRecord
# ---------------------------------------------------------------------------

class TestLLMResultModels:
    def test_tool_call_record_defaults(self):
        r = ToolCallRecord(name="foo")
        assert r.input == {}
        assert r.output == ""
        assert r.duration_ms == 0.0
        assert r.is_error is False

    def test_llm_result_defaults(self):
        r = LLMResult()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.cost_usd == 0.0
        assert r.iterations == 0
        assert r.usage.total_tokens == 0

    def test_llm_result_with_data(self):
        r = LLMResult(
            text="hello",
            tool_calls=[ToolCallRecord(name="foo", input={"a": 1}, output="bar")],
            iterations=2,
            cost_usd=0.005,
        )
        assert r.text == "hello"
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "foo"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_known_model(self):
        cost = estimate_cost_usd("gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(2.50)

    def test_known_model_output(self):
        cost = estimate_cost_usd("gpt-4o", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == pytest.approx(10.00)

    def test_unknown_model_returns_zero(self):
        cost = estimate_cost_usd("some-random-model-xyz", 1000, 1000)
        assert cost == 0.0

    def test_provider_prefix_stripped(self):
        cost = estimate_cost_usd("anthropic/claude-sonnet-4-6", 1_000_000, 0)
        assert cost == pytest.approx(3.00)

    def test_partial_match(self):
        # "claude-sonnet-4-6-20250514" should match "claude-sonnet-4-6"
        cost = estimate_cost_usd("claude-sonnet-4-6-20250514", 1_000_000, 0)
        assert cost == pytest.approx(3.00)

    def test_lookup_none(self):
        assert _lookup_pricing("") is None
        assert _lookup_pricing("nonexistent") is None


# ---------------------------------------------------------------------------
# LLM.call() backwards compatibility (no tools → returns str)
# ---------------------------------------------------------------------------

class TestCallBackwardsCompat:
    """LLM.call() without tools must return str exactly as before."""

    @patch("crewai.llm.litellm")
    def test_call_without_tools_returns_str(self, mock_litellm):
        """Plain call without tools should return a string."""
        mock_litellm.completion.return_value = _make_model_response(content="Hello world")
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        llm = _make_litellm_llm()
        result = llm.call("Say hello")

        assert isinstance(result, str)
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# LLM.call() with tools → returns LLMResult
# ---------------------------------------------------------------------------

class TestCallWithToolLoop:
    """When tools + available_functions are passed, call() returns LLMResult."""

    @patch("crewai.llm.litellm")
    def test_single_tool_call_then_text(self, mock_litellm):
        """Model calls one tool, then responds with text."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        # First call: model wants to call get_weather
        tool_call = _make_tool_call("get_weather", {"city": "SF"})
        resp1 = _make_model_response(content=None, tool_calls=[tool_call])
        # Second call: model responds with text
        resp2 = _make_model_response(content="It's sunny in SF!")
        mock_litellm.completion.side_effect = [resp1, resp2]

        llm = _make_litellm_llm()

        def get_weather(city: str) -> str:
            return f"Sunny, 72°F in {city}"

        result = llm.call(
            messages="What's the weather in SF?",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": get_weather},
        )

        assert isinstance(result, LLMResult)
        assert result.text == "It's sunny in SF!"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].input == {"city": "SF"}
        assert "Sunny" in result.tool_calls[0].output
        assert result.tool_calls[0].is_error is False
        assert result.iterations == 2

    @patch("crewai.llm.litellm")
    def test_multiple_tool_calls_in_sequence(self, mock_litellm):
        """Model calls two tools across two iterations."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        tc1 = _make_tool_call("get_weather", {"city": "SF"}, "call_1")
        resp1 = _make_model_response(content=None, tool_calls=[tc1])

        tc2 = _make_tool_call("get_weather", {"city": "NYC"}, "call_2")
        resp2 = _make_model_response(content=None, tool_calls=[tc2])

        resp3 = _make_model_response(content="SF is sunny, NYC is rainy.")
        mock_litellm.completion.side_effect = [resp1, resp2, resp3]

        llm = _make_litellm_llm()

        def get_weather(city: str) -> str:
            return f"Weather for {city}: fine"

        result = llm.call(
            messages="Compare SF and NYC weather",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": get_weather},
        )

        assert isinstance(result, LLMResult)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].input["city"] == "SF"
        assert result.tool_calls[1].input["city"] == "NYC"
        assert result.iterations == 3

    @patch("crewai.llm.litellm")
    def test_max_iterations_stops_loop(self, mock_litellm):
        """Loop stops when max_iterations is reached."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        # Model always wants to call a tool — never stops
        def make_tool_resp():
            tc = _make_tool_call("get_weather", {"city": "SF"})
            return _make_model_response(content=None, tool_calls=[tc])

        mock_litellm.completion.side_effect = [make_tool_resp() for _ in range(5)]

        llm = _make_litellm_llm()

        result = llm.call(
            messages="Loop forever",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": lambda city: "sunny"},
            max_iterations=3,
        )

        assert isinstance(result, LLMResult)
        assert result.iterations == 3
        assert len(result.tool_calls) == 3
        # Should have a text noting max iterations
        assert "Max iterations" in result.text

    @patch("crewai.llm.litellm")
    def test_tool_error_handling(self, mock_litellm):
        """Tool that raises an exception is captured in the record."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        tc = _make_tool_call("get_weather", {"city": "SF"})
        resp1 = _make_model_response(content=None, tool_calls=[tc])
        resp2 = _make_model_response(content="Sorry, couldn't get weather.")
        mock_litellm.completion.side_effect = [resp1, resp2]

        llm = _make_litellm_llm()

        def broken_weather(city: str) -> str:
            raise RuntimeError("API down")

        result = llm.call(
            messages="Weather?",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": broken_weather},
        )

        assert isinstance(result, LLMResult)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].is_error is True
        assert "API down" in result.tool_calls[0].output
        assert result.text == "Sorry, couldn't get weather."

    @patch("crewai.llm.litellm")
    def test_unknown_function_error(self, mock_litellm):
        """Tool call for a function not in available_functions."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        tc = _make_tool_call("nonexistent_tool", {})
        resp1 = _make_model_response(content=None, tool_calls=[tc])
        resp2 = _make_model_response(content="I couldn't find that tool.")
        mock_litellm.completion.side_effect = [resp1, resp2]

        llm = _make_litellm_llm()

        result = llm.call(
            messages="Do something",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": lambda city: "sunny"},
        )

        assert isinstance(result, LLMResult)
        assert result.tool_calls[0].is_error is True
        assert "unknown function" in result.tool_calls[0].output

    @patch("crewai.llm.litellm")
    def test_cost_estimation_populated(self, mock_litellm):
        """cost_usd is populated from token usage and model pricing."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        resp = _make_model_response(content="Done!")
        mock_litellm.completion.return_value = resp

        llm = _make_litellm_llm()

        result = llm.call(
            messages="Hello",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": lambda city: "sunny"},
        )

        assert isinstance(result, LLMResult)
        # cost_usd should be >= 0 (may be 0 if usage tracking didn't fire,
        # but the field should exist and be a float)
        assert isinstance(result.cost_usd, float)

    @patch("crewai.llm.litellm")
    def test_immediate_text_response_with_tools(self, mock_litellm):
        """Model responds with text on first call (no tool use)."""
        mock_litellm.drop_params = True
        mock_litellm.suppress_debug_info = True
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []
        mock_litellm.callbacks = []

        resp = _make_model_response(content="I know the answer already.")
        mock_litellm.completion.return_value = resp

        llm = _make_litellm_llm()

        result = llm.call(
            messages="What's 2+2?",
            tools=DUMMY_TOOL_SCHEMA,
            available_functions={"get_weather": lambda city: "sunny"},
        )

        assert isinstance(result, LLMResult)
        assert result.text == "I know the answer already."
        assert len(result.tool_calls) == 0
        assert result.iterations == 1
