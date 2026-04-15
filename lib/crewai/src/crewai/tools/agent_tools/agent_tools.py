from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
from crewai.utilities.i18n import I18N_DEFAULT


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


class AgentTools:
    """Manager class for agent-related tools"""

    def __init__(self, agents: Sequence[BaseAgent], task: Task | None = None) -> None:
        self.agents = agents
        self.task = task

    def tools(self) -> list[BaseTool]:
        """Get all available agent tools.

        When a task is provided, its constraints are automatically propagated
        to the delegation tools so that worker agents receive them.
        """
        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])

        delegate_tool = DelegateWorkTool(
            agents=self.agents,
            original_task=self.task,
            description=I18N_DEFAULT.tools("delegate_work").format(coworkers=coworkers),  # type: ignore
        )

        ask_tool = AskQuestionTool(
            agents=self.agents,
            original_task=self.task,
            description=I18N_DEFAULT.tools("ask_question").format(coworkers=coworkers),  # type: ignore
        )

        return [delegate_tool, ask_tool]
