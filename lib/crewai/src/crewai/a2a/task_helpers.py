"""Helper functions for processing A2A task results."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from typing_extensions import NotRequired, TypedDict

from crewai.a2a._compat import (
    ROLE_AGENT,
    TASK_STATE_AUTH_REQUIRED,
    TASK_STATE_CANCELED,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
    TASK_STATE_INPUT_REQUIRED,
    TASK_STATE_REJECTED,
    TASK_STATE_SUBMITTED,
    TASK_STATE_WORKING,
    agent_card_to_dict,
    is_stream_artifact_update,
    is_stream_message,
    is_stream_status_update,
    is_stream_task,
    new_text_message,
    new_text_part,
    part_is_text,
    part_text,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConnectionErrorEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask

SendMessageEvent = StreamResponse


TERMINAL_STATES: frozenset[int] = frozenset(
    {
        TASK_STATE_COMPLETED,
        TASK_STATE_FAILED,
        TASK_STATE_REJECTED,
        TASK_STATE_CANCELED,
    }
)

ACTIONABLE_STATES: frozenset[int] = frozenset(
    {
        TASK_STATE_INPUT_REQUIRED,
        TASK_STATE_AUTH_REQUIRED,
    }
)

PENDING_STATES: frozenset[int] = frozenset(
    {
        TASK_STATE_SUBMITTED,
        TASK_STATE_WORKING,
    }
)


class TaskStateResult(TypedDict):
    """Result dictionary from processing A2A task state."""

    status: int
    history: list[Message]
    result: NotRequired[str]
    error: NotRequired[str]
    agent_card: NotRequired[dict[str, Any]]
    a2a_agent_name: NotRequired[str | None]


def extract_task_result_parts(a2a_task: A2ATask) -> list[str]:
    """Extract result parts from A2A task status message, history, and artifacts.

    Args:
        a2a_task: A2A Task object with status, history, and artifacts

    Returns:
        List of result text parts
    """
    result_parts: list[str] = []

    if a2a_task.status and a2a_task.status.message:
        msg = a2a_task.status.message
        result_parts.extend(
            part_text(part) for part in msg.parts if part_is_text(part)
        )

    if not result_parts and a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            if history_msg.role == ROLE_AGENT:
                result_parts.extend(
                    part_text(part)
                    for part in history_msg.parts
                    if part_is_text(part)
                )
                break

    if a2a_task.artifacts:
        result_parts.extend(
            part_text(part)
            for artifact in a2a_task.artifacts
            for part in artifact.parts
            if part_is_text(part)
        )

    return result_parts


def extract_error_message(a2a_task: A2ATask, default: str) -> str:
    """Extract error message from A2A task.

    Args:
        a2a_task: A2A Task object
        default: Default message if no error found

    Returns:
        Error message string
    """
    if a2a_task.status and a2a_task.status.message:
        msg = a2a_task.status.message
        if msg:
            for part in msg.parts:
                if part_is_text(part):
                    return str(part_text(part))
            return str(msg)

    if a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            for part in history_msg.parts:
                if part_is_text(part):
                    return str(part_text(part))

    return default


def process_task_state(
    a2a_task: A2ATask,
    new_messages: list[Message],
    agent_card: AgentCard,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    result_parts: list[str] | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    is_final: bool = True,
) -> TaskStateResult | None:
    """Process A2A task state and return result dictionary.

    Shared logic for both polling and streaming handlers.

    Args:
        a2a_task: The A2A task to process.
        new_messages: List to collect messages (modified in place).
        agent_card: The agent card.
        turn_number: Current turn number.
        is_multiturn: Whether multi-turn conversation.
        agent_role: Agent role for logging.
        result_parts: Accumulated result parts (streaming passes accumulated,
            polling passes None to extract from task).
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        from_task: Optional CrewAI Task for event metadata.
        from_agent: Optional CrewAI Agent for event metadata.
        is_final: Whether this is the final response in the stream.

    Returns:
        Result dictionary if terminal/actionable state, None otherwise.
    """
    if result_parts is None:
        result_parts = []

    if a2a_task.status.state == TASK_STATE_COMPLETED:
        if not result_parts:
            extracted_parts = extract_task_result_parts(a2a_task)
            result_parts.extend(extracted_parts)
        if a2a_task.history:
            new_messages.extend(a2a_task.history)

        response_text = " ".join(result_parts) if result_parts else ""
        message_id = None
        if a2a_task.status and a2a_task.status.message:
            message_id = a2a_task.status.message.message_id
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=response_text,
                turn_number=turn_number,
                context_id=a2a_task.context_id,
                message_id=message_id,
                is_multiturn=is_multiturn,
                status="completed",
                final=is_final,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        return TaskStateResult(
            status=TASK_STATE_COMPLETED,
            agent_card=agent_card_to_dict(agent_card),
            result=response_text,
            history=new_messages,
        )

    if a2a_task.status.state == TASK_STATE_INPUT_REQUIRED:
        if a2a_task.history:
            new_messages.extend(a2a_task.history)

        response_text = extract_error_message(a2a_task, "Additional input required")
        if response_text and not a2a_task.history:
            agent_message = new_text_message(
                response_text,
                role=ROLE_AGENT,
                context_id=a2a_task.context_id,
                task_id=a2a_task.id,
            )
            new_messages.append(agent_message)

        input_message_id = None
        if a2a_task.status and a2a_task.status.message:
            input_message_id = a2a_task.status.message.message_id
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=response_text,
                turn_number=turn_number,
                context_id=a2a_task.context_id,
                message_id=input_message_id,
                is_multiturn=is_multiturn,
                status="input_required",
                final=is_final,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        return TaskStateResult(
            status=TASK_STATE_INPUT_REQUIRED,
            error=response_text,
            history=new_messages,
            agent_card=agent_card_to_dict(agent_card),
        )

    if a2a_task.status.state in {TASK_STATE_FAILED, TASK_STATE_REJECTED}:
        error_msg = extract_error_message(a2a_task, "Task failed without error message")
        if a2a_task.history:
            new_messages.extend(a2a_task.history)
        return TaskStateResult(
            status=TASK_STATE_FAILED,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state == TASK_STATE_AUTH_REQUIRED:
        error_msg = extract_error_message(a2a_task, "Authentication required")
        return TaskStateResult(
            status=TASK_STATE_AUTH_REQUIRED,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state == TASK_STATE_CANCELED:
        error_msg = extract_error_message(a2a_task, "Task was canceled")
        return TaskStateResult(
            status=TASK_STATE_CANCELED,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state in PENDING_STATES:
        return None

    return None


async def send_message_and_get_task_id(
    event_stream: AsyncIterator[StreamResponse],
    new_messages: list[Message],
    agent_card: AgentCard,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    context_id: str | None = None,
) -> str | TaskStateResult:
    """Send message and process initial response.

    Handles the common pattern of sending a message and either:
    - Getting an immediate Message response (task completed synchronously)
    - Getting a Task that needs polling/waiting for completion

    Args:
        event_stream: Async iterator from client.send_message()
        new_messages: List to collect messages (modified in place)
        agent_card: The agent card
        turn_number: Current turn number
        is_multiturn: Whether multi-turn conversation
        agent_role: Agent role for logging
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        endpoint: Optional A2A endpoint URL.
        a2a_agent_name: Optional A2A agent name.
        context_id: Optional A2A context ID for correlation.

    Returns:
        Task ID string if agent needs polling/waiting, or TaskStateResult if done.
    """
    try:
        async for chunk in event_stream:
            if is_stream_message(chunk):
                event = chunk.message
                new_messages.append(event)
                result_parts = [
                    part_text(part) for part in event.parts if part_is_text(part)
                ]
                response_text = " ".join(result_parts) if result_parts else ""

                crewai_event_bus.emit(
                    None,
                    A2AResponseReceivedEvent(
                        response=response_text,
                        turn_number=turn_number,
                        context_id=event.context_id,
                        message_id=event.message_id,
                        is_multiturn=is_multiturn,
                        status="completed",
                        final=True,
                        agent_role=agent_role,
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                return TaskStateResult(
                    status=TASK_STATE_COMPLETED,
                    result=response_text,
                    history=new_messages,
                    agent_card=agent_card_to_dict(agent_card),
                )

            if is_stream_task(chunk):
                a2a_task = chunk.task

                if a2a_task.status.state in TERMINAL_STATES | ACTIONABLE_STATES:
                    result = process_task_state(
                        a2a_task=a2a_task,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        turn_number=turn_number,
                        is_multiturn=is_multiturn,
                        agent_role=agent_role,
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        from_task=from_task,
                        from_agent=from_agent,
                    )
                    if result:
                        return result

                return a2a_task.id

        return TaskStateResult(
            status=TASK_STATE_FAILED,
            error="No task ID received from initial message",
            history=new_messages,
        )

    except A2AClientError as e:
        error_msg = f"A2A Client Error: {e!s}"

        error_message = new_text_message(
            error_msg,
            role=ROLE_AGENT,
            context_id=context_id,
        )
        new_messages.append(error_message)

        crewai_event_bus.emit(
            None,
            A2AConnectionErrorEvent(
                endpoint=endpoint or "",
                error=str(e),
                error_type="client_error",
                a2a_agent_name=a2a_agent_name,
                operation="send_message",
                context_id=context_id,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=error_msg,
                turn_number=turn_number,
                context_id=context_id,
                is_multiturn=is_multiturn,
                status="failed",
                final=True,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        return TaskStateResult(
            status=TASK_STATE_FAILED,
            error=error_msg,
            history=new_messages,
        )

    except Exception as e:
        error_msg = f"Unexpected error during send_message: {e!s}"

        error_message = new_text_message(
            error_msg,
            role=ROLE_AGENT,
            context_id=context_id,
        )
        new_messages.append(error_message)

        crewai_event_bus.emit(
            None,
            A2AConnectionErrorEvent(
                endpoint=endpoint or "",
                error=str(e),
                error_type="unexpected_error",
                a2a_agent_name=a2a_agent_name,
                operation="send_message",
                context_id=context_id,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=error_msg,
                turn_number=turn_number,
                context_id=context_id,
                is_multiturn=is_multiturn,
                status="failed",
                final=True,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        return TaskStateResult(
            status=TASK_STATE_FAILED,
            error=error_msg,
            history=new_messages,
        )

    finally:
        aclose = getattr(event_stream, "aclose", None)
        if aclose:
            await aclose()
