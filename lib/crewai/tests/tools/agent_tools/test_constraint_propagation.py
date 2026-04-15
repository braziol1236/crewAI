"""Tests for constraint propagation during task delegation.

These tests verify that when a Task has structured constraints defined,
they are properly propagated to delegated tasks through the DelegateWorkTool
and AskQuestionTool, ensuring worker agents receive the original requirements.

See: https://github.com/crewAIInc/crewAI/issues/5476
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.task import Task
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.agent_tools.base_agent_tools import BaseAgentTool
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool


@pytest.fixture
def researcher():
    return Agent(
        role="researcher",
        goal="Research AI topics",
        backstory="Expert researcher in AI",
        allow_delegation=False,
    )


@pytest.fixture
def writer():
    return Agent(
        role="writer",
        goal="Write articles about AI",
        backstory="Expert technical writer",
        allow_delegation=False,
    )


@pytest.fixture
def task_with_constraints(researcher):
    return Task(
        description="Find the best open-source ML frameworks from 2024 in Europe",
        expected_output="A list of ML frameworks",
        agent=researcher,
        constraints=[
            "Only open-source frameworks",
            "Must be from 2024",
            "Only frameworks available in Europe",
        ],
    )


@pytest.fixture
def task_without_constraints(researcher):
    return Task(
        description="Find ML frameworks",
        expected_output="A list of ML frameworks",
        agent=researcher,
    )


class TestTaskConstraintsField:
    """Tests for the constraints field on the Task model."""

    def test_task_has_constraints_field(self):
        """A Task can be created with a constraints field."""
        task = Task(
            description="Test task",
            expected_output="Test output",
            constraints=["constraint1", "constraint2"],
        )
        assert task.constraints == ["constraint1", "constraint2"]

    def test_task_constraints_default_empty(self):
        """A Task without constraints has an empty list by default."""
        task = Task(
            description="Test task",
            expected_output="Test output",
        )
        assert task.constraints == []

    def test_task_prompt_includes_constraints(self):
        """Task.prompt() includes constraints when they are set."""
        task = Task(
            description="Find ML frameworks",
            expected_output="A list of frameworks",
            constraints=["Only open-source", "From 2024 only"],
        )
        prompt = task.prompt()
        assert "Task Constraints (MUST be respected):" in prompt
        assert "- Only open-source" in prompt
        assert "- From 2024 only" in prompt

    def test_task_prompt_excludes_constraints_when_empty(self):
        """Task.prompt() does not include constraint section when constraints are empty."""
        task = Task(
            description="Find ML frameworks",
            expected_output="A list of frameworks",
        )
        prompt = task.prompt()
        assert "Task Constraints" not in prompt


class TestConstraintPropagationInDelegation:
    """Tests for constraint propagation through delegation tools."""

    def test_delegate_tool_receives_original_task(self, researcher, writer, task_with_constraints):
        """DelegateWorkTool is initialized with the original task reference."""
        tools = AgentTools(agents=[writer], task=task_with_constraints).tools()
        delegate_tool = tools[0]
        assert isinstance(delegate_tool, DelegateWorkTool)
        assert delegate_tool.original_task is task_with_constraints

    def test_ask_tool_receives_original_task(self, researcher, writer, task_with_constraints):
        """AskQuestionTool is initialized with the original task reference."""
        tools = AgentTools(agents=[writer], task=task_with_constraints).tools()
        ask_tool = tools[1]
        assert isinstance(ask_tool, AskQuestionTool)
        assert ask_tool.original_task is task_with_constraints

    def test_delegate_tool_without_task_has_none(self, writer):
        """When no task is provided, original_task is None."""
        tools = AgentTools(agents=[writer]).tools()
        delegate_tool = tools[0]
        assert delegate_tool.original_task is None

    @patch.object(Agent, "execute_task")
    def test_constraints_propagated_to_delegated_task(
        self, mock_execute, researcher, writer, task_with_constraints
    ):
        """Constraints from the original task are propagated to the delegated task."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_with_constraints).tools()
        delegate_tool = tools[0]

        delegate_tool.run(
            coworker="researcher",
            task="Find ML frameworks",
            context="Need a comprehensive list",
        )

        # Verify execute_task was called
        mock_execute.assert_called_once()
        delegated_task = mock_execute.call_args[0][0]
        delegated_context = mock_execute.call_args[0][1]

        # The delegated task should have the constraints from the original task
        assert delegated_task.constraints == [
            "Only open-source frameworks",
            "Must be from 2024",
            "Only frameworks available in Europe",
        ]

        # The context should include the constraints
        assert "Task Constraints (MUST be respected):" in delegated_context
        assert "- Only open-source frameworks" in delegated_context
        assert "- Must be from 2024" in delegated_context
        assert "- Only frameworks available in Europe" in delegated_context

    @patch.object(Agent, "execute_task")
    def test_constraints_appended_to_existing_context(
        self, mock_execute, researcher, writer, task_with_constraints
    ):
        """When context already exists, constraints are appended to it."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_with_constraints).tools()
        delegate_tool = tools[0]

        delegate_tool.run(
            coworker="researcher",
            task="Find ML frameworks",
            context="Previous context here",
        )

        mock_execute.assert_called_once()
        delegated_context = mock_execute.call_args[0][1]

        # Original context should still be there
        assert delegated_context.startswith("Previous context here")
        # Constraints should be appended
        assert "Task Constraints (MUST be respected):" in delegated_context

    @patch.object(Agent, "execute_task")
    def test_no_constraints_no_modification(
        self, mock_execute, researcher, writer, task_without_constraints
    ):
        """When original task has no constraints, context is not modified."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_without_constraints).tools()
        delegate_tool = tools[0]

        delegate_tool.run(
            coworker="researcher",
            task="Find ML frameworks",
            context="Just context",
        )

        mock_execute.assert_called_once()
        delegated_task = mock_execute.call_args[0][0]
        delegated_context = mock_execute.call_args[0][1]

        assert delegated_task.constraints == []
        assert delegated_context == "Just context"

    @patch.object(Agent, "execute_task")
    def test_ask_question_propagates_constraints(
        self, mock_execute, researcher, writer, task_with_constraints
    ):
        """AskQuestionTool also propagates constraints to the delegated task."""
        mock_execute.return_value = "answer"

        tools = AgentTools(agents=[researcher], task=task_with_constraints).tools()
        ask_tool = tools[1]

        ask_tool.run(
            coworker="researcher",
            question="What are the best frameworks?",
            context="Need details",
        )

        mock_execute.assert_called_once()
        delegated_task = mock_execute.call_args[0][0]
        delegated_context = mock_execute.call_args[0][1]

        assert delegated_task.constraints == task_with_constraints.constraints
        assert "Task Constraints (MUST be respected):" in delegated_context

    @patch.object(Agent, "execute_task")
    def test_constraints_propagated_when_no_original_context(
        self, mock_execute, researcher, writer, task_with_constraints
    ):
        """When delegation has no context, constraints become the context."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_with_constraints).tools()
        delegate_tool = tools[0]

        delegate_tool.run(
            coworker="researcher",
            task="Find ML frameworks",
            context="",
        )

        mock_execute.assert_called_once()
        delegated_context = mock_execute.call_args[0][1]

        # Empty string context means constraints text is appended to empty string
        assert "Task Constraints (MUST be respected):" in delegated_context

    @patch.object(Agent, "execute_task")
    def test_delegation_without_original_task_works(
        self, mock_execute, researcher, writer
    ):
        """Delegation still works when no original task is set (backward compatible)."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher]).tools()
        delegate_tool = tools[0]

        delegate_tool.run(
            coworker="researcher",
            task="Find ML frameworks",
            context="Some context",
        )

        mock_execute.assert_called_once()
        delegated_task = mock_execute.call_args[0][0]
        delegated_context = mock_execute.call_args[0][1]

        # Should work normally without constraints
        assert delegated_task.constraints == []
        assert delegated_context == "Some context"


class TestConstraintPropagationLogging:
    """Tests for logging during constraint propagation."""

    @patch.object(Agent, "execute_task")
    def test_constraint_propagation_logs_info(
        self, mock_execute, researcher, writer, task_with_constraints, caplog
    ):
        """An info log is emitted when constraints are propagated."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_with_constraints).tools()
        delegate_tool = tools[0]

        with caplog.at_level(logging.INFO, logger="crewai.tools.agent_tools.base_agent_tools"):
            delegate_tool.run(
                coworker="researcher",
                task="Find ML frameworks",
                context="Context",
            )

        assert any("Propagating 3 constraint(s)" in record.message for record in caplog.records)

    @patch.object(Agent, "execute_task")
    def test_no_log_when_no_constraints(
        self, mock_execute, researcher, writer, task_without_constraints, caplog
    ):
        """No constraint propagation log when there are no constraints."""
        mock_execute.return_value = "result"

        tools = AgentTools(agents=[researcher], task=task_without_constraints).tools()
        delegate_tool = tools[0]

        with caplog.at_level(logging.INFO, logger="crewai.tools.agent_tools.base_agent_tools"):
            delegate_tool.run(
                coworker="researcher",
                task="Find ML frameworks",
                context="Context",
            )

        assert not any("Propagating" in record.message for record in caplog.records)


class TestAgentToolsTaskPassThrough:
    """Tests that AgentTools passes the task to the underlying tools."""

    def test_agent_tools_with_task(self, researcher, task_with_constraints):
        """AgentTools passes the task to both delegate and ask tools."""
        agent_tools = AgentTools(agents=[researcher], task=task_with_constraints)
        tools = agent_tools.tools()

        assert len(tools) == 2
        for tool in tools:
            assert isinstance(tool, BaseAgentTool)
            assert tool.original_task is task_with_constraints

    def test_agent_tools_without_task(self, researcher):
        """AgentTools without a task sets original_task to None on tools."""
        agent_tools = AgentTools(agents=[researcher])
        tools = agent_tools.tools()

        assert len(tools) == 2
        for tool in tools:
            assert isinstance(tool, BaseAgentTool)
            assert tool.original_task is None

    def test_agent_get_delegation_tools_passes_task(self, researcher, task_with_constraints):
        """Agent.get_delegation_tools passes the task through to AgentTools."""
        tools = researcher.get_delegation_tools(agents=[researcher], task=task_with_constraints)

        assert len(tools) == 2
        for tool in tools:
            assert isinstance(tool, BaseAgentTool)
            assert tool.original_task is task_with_constraints

    def test_agent_get_delegation_tools_without_task(self, researcher):
        """Agent.get_delegation_tools without task still works (backward compatible)."""
        tools = researcher.get_delegation_tools(agents=[researcher])

        assert len(tools) == 2
        for tool in tools:
            assert isinstance(tool, BaseAgentTool)
            assert tool.original_task is None
