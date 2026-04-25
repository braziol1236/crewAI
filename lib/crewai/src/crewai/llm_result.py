"""Structured result types for LLM.call() with tool loop support.

When LLM.call() is invoked with tools and available_functions, it returns
an LLMResult instead of a plain string. This preserves backwards compatibility:
calls without tools still return str.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.types.usage_metrics import UsageMetrics


class ToolCallRecord(BaseModel):
    """Record of a single tool call executed during an LLM tool loop.

    Attributes:
        name: The tool function name.
        input: The arguments passed to the tool.
        output: The string result returned by the tool.
        duration_ms: Wall-clock time for the tool execution in milliseconds.
        is_error: Whether the tool call raised an exception.
    """

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    duration_ms: float = 0.0
    is_error: bool = False


class LLMResult(BaseModel):
    """Structured result from LLM.call() when tools are used.

    Attributes:
        text: The final text response from the model.
        tool_calls: Ordered list of every tool call made during the loop.
        usage: Aggregated token usage across all iterations.
        cost_usd: Estimated cost in USD based on model pricing.
        iterations: Number of LLM round-trips in the tool loop.
    """

    text: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    cost_usd: float = 0.0
    iterations: int = 0


# ---------------------------------------------------------------------------
# Simple cost estimation
# ---------------------------------------------------------------------------
# USD per 1M tokens. Covers major models. Inspired by Iris's pricing table.
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7": {"in": 5.00, "out": 25.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-sonnet-4-5": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    # OpenAI
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
    "o1": {"in": 15.00, "out": 60.00},
    "o1-mini": {"in": 3.00, "out": 12.00},
    "o3": {"in": 2.00, "out": 8.00},
    "o3-mini": {"in": 1.10, "out": 4.40},
    "gpt-5": {"in": 1.25, "out": 10.00},
    # Google Gemini
    "gemini-2.5-pro": {"in": 1.25, "out": 10.00},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
}


def _lookup_pricing(model: str) -> dict[str, float] | None:
    """Resolve a model name to its pricing row.

    Handles provider prefixes (``anthropic/claude-sonnet-4-6``) and partial
    matches (``claude-sonnet-4-6-20250514`` → ``claude-sonnet-4-6``).
    """
    if not model:
        return None
    # Exact match
    if model in PRICING:
        return PRICING[model]
    # Strip provider prefix
    if "/" in model:
        suffix = model.rsplit("/", 1)[1]
        if suffix in PRICING:
            return PRICING[suffix]
        model = suffix
    # Prefix / partial match
    for key in PRICING:
        if model.startswith(key) or key.startswith(model):
            return PRICING[key]
    return None


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate the cost in USD for a given model and token counts."""
    pricing = _lookup_pricing(model)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["in"] + completion_tokens * pricing["out"]) / 1_000_000
