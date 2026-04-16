"""Tests for the crewai.cli.crew_chat description generators.

These tests focus on the defensive behaviour introduced for issue #5510:
``generate_input_description_with_ai`` and ``generate_crew_description_with_ai``
must never propagate LLM call failures to their callers, since they are
commonly invoked at container / module import time via downstream
integrations such as ``ag_ui_crewai.crews.ChatWithCrewFlow``. A transient LLM
provider hiccup should not crash the containing process before it has a chance
to bind to its HTTP port.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai.agent import Agent
from crewai.cli.crew_chat import (
    DEFAULT_CREW_DESCRIPTION,
    DEFAULT_INPUT_DESCRIPTION,
    generate_crew_chat_inputs,
    generate_crew_description_with_ai,
    generate_input_description_with_ai,
)
from crewai.crew import Crew
from crewai.task import Task


def _make_crew_with_topic_input() -> Crew:
    """Build a minimal Crew whose task/agent reference a ``{topic}`` input."""
    agent = Agent(
        role="Researcher on {topic}",
        goal="Investigate the latest developments about {topic}",
        backstory="An expert analyst focused on {topic}",
        allow_delegation=False,
    )
    task = Task(
        description="Write a short report about {topic}",
        expected_output="A concise summary about {topic}",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


def test_generate_input_description_returns_llm_response_on_success() -> None:
    """Happy path: the LLM response is stripped and returned verbatim."""
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()
    chat_llm.call.return_value = "  The topic to research.  "

    result = generate_input_description_with_ai("topic", crew, chat_llm)

    assert result == "The topic to research."
    chat_llm.call.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection refused"),
        TimeoutError("llm timed out"),
        RuntimeError("litellm APIError: 500"),
    ],
)
def test_generate_input_description_falls_back_on_llm_failure(exc: Exception) -> None:
    """If the LLM call raises, we must return the static fallback instead of
    propagating the exception. This is the core fix for issue #5510.
    """
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()
    chat_llm.call.side_effect = exc

    result = generate_input_description_with_ai("topic", crew, chat_llm)

    assert result == DEFAULT_INPUT_DESCRIPTION


def test_generate_input_description_still_raises_when_no_context() -> None:
    """The fallback only applies to LLM call failures. When there is no
    context at all for the given input, we still raise ``ValueError`` so that
    callers can detect a truly malformed crew definition.
    """
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()

    with pytest.raises(ValueError, match="No context found for input"):
        generate_input_description_with_ai("does_not_exist", crew, chat_llm)

    chat_llm.call.assert_not_called()


def test_generate_crew_description_returns_llm_response_on_success() -> None:
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()
    chat_llm.call.return_value = "  Research topics and produce reports.  "

    result = generate_crew_description_with_ai(crew, chat_llm)

    assert result == "Research topics and produce reports."
    chat_llm.call.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection refused"),
        TimeoutError("llm timed out"),
        RuntimeError("litellm APIError: 500"),
    ],
)
def test_generate_crew_description_falls_back_on_llm_failure(exc: Exception) -> None:
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()
    chat_llm.call.side_effect = exc

    result = generate_crew_description_with_ai(crew, chat_llm)

    assert result == DEFAULT_CREW_DESCRIPTION


def test_generate_crew_chat_inputs_never_crashes_on_llm_failure() -> None:
    """End-to-end: a crew with at least one required input placeholder and a
    chat LLM whose ``.call`` always raises should still yield a valid
    ``ChatInputs`` object populated with the static fallbacks, rather than
    bubbling up the exception. This is the exact scenario described in
    issue #5510 for ``ChatWithCrewFlow.__init__``.
    """
    crew = _make_crew_with_topic_input()
    chat_llm = MagicMock()
    chat_llm.call.side_effect = ConnectionError("transient outage")

    chat_inputs = generate_crew_chat_inputs(crew, "MyCrew", chat_llm)

    assert chat_inputs.crew_name == "MyCrew"
    assert chat_inputs.crew_description == DEFAULT_CREW_DESCRIPTION
    assert len(chat_inputs.inputs) == 1
    assert chat_inputs.inputs[0].name == "topic"
    assert chat_inputs.inputs[0].description == DEFAULT_INPUT_DESCRIPTION
