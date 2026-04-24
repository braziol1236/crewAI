"""Streaming (SSE) update mechanism handler."""

from __future__ import annotations

import asyncio
import logging
from typing import Final
import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    Part,
    Role,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from typing_extensions import Unpack

from crewai.a2a._compat import (
    ROLE_AGENT,
    TASK_STATE_FAILED,
    agent_card_to_dict,
    is_status_update_final,
    is_stream_artifact_update,
    is_stream_message,
    is_stream_status_update,
    is_stream_task,
    new_text_message,
    part_is_text,
    part_text,
)
from crewai.a2a.task_helpers import (
    ACTIONABLE_STATES,
    TERMINAL_STATES,
    TaskStateResult,
    process_task_state,
)
from crewai.a2a.updates.base import StreamingHandlerKwargs, extract_common_params
from crewai.a2a.updates.streaming.params import (
    process_status_update,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AArtifactReceivedEvent,
    A2AConnectionErrorEvent,
    A2AResponseReceivedEvent,
    A2AStreamingChunkEvent,
    A2AStreamingStartedEvent,
)


logger = logging.getLogger(__name__)

MAX_RESUBSCRIBE_ATTEMPTS: Final[int] = 3
RESUBSCRIBE_BACKOFF_BASE: Final[float] = 1.0


def _extract_text_from_artifact(artifact: TaskArtifactUpdateEvent) -> list[str]:
    """Extract text parts from an artifact update event."""
    parts: list[str] = []
    if artifact.artifact and artifact.artifact.parts:
        parts.extend(
            part_text(part)
            for part in artifact.artifact.parts
            if part_is_text(part)
        )
    return parts


class StreamingHandler:
    """SSE streaming-based update handler."""

    @staticmethod
    async def _try_recover_from_interruption(  # type: ignore[misc]
        client: Client,
        task_id: str,
        new_messages: list[Message],
        agent_card: AgentCard,
        result_parts: list[str],
        **kwargs: Unpack[StreamingHandlerKwargs],
    ) -> TaskStateResult | None:
        """Attempt to recover from a stream interruption by checking task state.

        If the task completed while we were disconnected, returns the result.
        If the task is still running, attempts to resubscribe and continue.

        Args:
            client: A2A client instance.
            task_id: The task ID to recover.
            new_messages: List of collected messages.
            agent_card: The agent card.
            result_parts: Accumulated result text parts.
            **kwargs: Handler parameters.

        Returns:
            TaskStateResult if recovery succeeded (task finished or resubscribe worked).
            None if recovery not possible (caller should handle failure).

        Note:
            When None is returned, recovery failed and the original exception should
            be handled by the caller. All recovery attempts are logged.
        """
        params = extract_common_params(kwargs)  # type: ignore[arg-type]

        try:
            a2a_task: Task = await client.get_task(GetTaskRequest(id=task_id))

            if a2a_task.status.state in TERMINAL_STATES:
                logger.info(
                    "Task completed during stream interruption",
                    extra={"task_id": task_id, "state": str(a2a_task.status.state)},
                )
                return process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=params.turn_number,
                    is_multiturn=params.is_multiturn,
                    agent_role=params.agent_role,
                    result_parts=result_parts,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                )

            if a2a_task.status.state in ACTIONABLE_STATES:
                logger.info(
                    "Task in actionable state during stream interruption",
                    extra={"task_id": task_id, "state": str(a2a_task.status.state)},
                )
                return process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=params.turn_number,
                    is_multiturn=params.is_multiturn,
                    agent_role=params.agent_role,
                    result_parts=result_parts,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                    is_final=False,
                )

            logger.info(
                "Task still running, attempting resubscribe",
                extra={"task_id": task_id, "state": str(a2a_task.status.state)},
            )

            for attempt in range(MAX_RESUBSCRIBE_ATTEMPTS):
                try:
                    backoff = RESUBSCRIBE_BACKOFF_BASE * (2**attempt)
                    if attempt > 0:
                        await asyncio.sleep(backoff)

                    event_stream = client.subscribe(SubscribeToTaskRequest(id=task_id))

                    async for chunk in event_stream:
                        if is_stream_task(chunk):
                            resubscribed_task = chunk.task

                        if is_stream_status_update(chunk):
                            update = chunk.status_update
                            is_final_update = process_status_update(update, result_parts)

                            if (
                                is_final_update
                                or resubscribed_task.status.state
                                in TERMINAL_STATES | ACTIONABLE_STATES
                            ):
                                return process_task_state(
                                    a2a_task=resubscribed_task,
                                    new_messages=new_messages,
                                    agent_card=agent_card,
                                    turn_number=params.turn_number,
                                    is_multiturn=params.is_multiturn,
                                    agent_role=params.agent_role,
                                    result_parts=result_parts,
                                    endpoint=params.endpoint,
                                    a2a_agent_name=params.a2a_agent_name,
                                    from_task=params.from_task,
                                    from_agent=params.from_agent,
                                    is_final=is_final_update,
                                )

                        if is_stream_artifact_update(chunk):
                            artifact = chunk.artifact_update
                            result_parts.extend(_extract_text_from_artifact(artifact))

                        if is_stream_message(chunk):
                            msg = chunk.message
                            new_messages.append(msg)
                            result_parts.extend(
                                part_text(part)
                                for part in msg.parts
                                if part_is_text(part)
                            )

                    final_task = await client.get_task(GetTaskRequest(id=task_id))
                    return process_task_state(
                        a2a_task=final_task,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        turn_number=params.turn_number,
                        is_multiturn=params.is_multiturn,
                        agent_role=params.agent_role,
                        result_parts=result_parts,
                        endpoint=params.endpoint,
                        a2a_agent_name=params.a2a_agent_name,
                        from_task=params.from_task,
                        from_agent=params.from_agent,
                    )

                except Exception as resubscribe_error:  # noqa: PERF203
                    logger.warning(
                        "Resubscribe attempt failed",
                        extra={
                            "task_id": task_id,
                            "attempt": attempt + 1,
                            "max_attempts": MAX_RESUBSCRIBE_ATTEMPTS,
                            "error": str(resubscribe_error),
                        },
                    )
                    if attempt == MAX_RESUBSCRIBE_ATTEMPTS - 1:
                        return None

        except Exception as e:
            logger.warning(
                "Failed to recover from stream interruption due to unexpected error",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return None

        logger.warning(
            "Recovery exhausted all resubscribe attempts without success",
            extra={"task_id": task_id, "max_attempts": MAX_RESUBSCRIBE_ATTEMPTS},
        )
        return None

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[StreamingHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using SSE streaming for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Streaming-specific parameters.

        Returns:
            Dictionary with status, result/error, and history.
        """
        task_id = kwargs.get("task_id")
        agent_branch = kwargs.get("agent_branch")
        params = extract_common_params(kwargs)

        result_parts: list[str] = []
        final_result: TaskStateResult | None = None
        from crewai.a2a._compat import make_send_request

        event_stream = client.send_message(make_send_request(message))
        chunk_index = 0
        current_task_id: str | None = task_id
        current_task: Task | None = None

        crewai_event_bus.emit(
            agent_branch,
            A2AStreamingStartedEvent(
                task_id=task_id,
                context_id=params.context_id,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
                turn_number=params.turn_number,
                is_multiturn=params.is_multiturn,
                agent_role=params.agent_role,
                from_task=params.from_task,
                from_agent=params.from_agent,
            ),
        )

        try:
            async for chunk in event_stream:
                # Extract task from task payload
                if is_stream_task(chunk):
                    current_task = chunk.task
                    current_task_id = current_task.id

                # Handle standalone message responses
                if is_stream_message(chunk):
                    msg = chunk.message
                    new_messages.append(msg)
                    message_context_id = msg.context_id or params.context_id
                    for part in msg.parts:
                        if part_is_text(part):
                            text = part_text(part)
                            result_parts.append(text)
                            crewai_event_bus.emit(
                                agent_branch,
                                A2AStreamingChunkEvent(
                                    task_id=msg.task_id or task_id,
                                    context_id=message_context_id,
                                    chunk=text,
                                    chunk_index=chunk_index,
                                    endpoint=params.endpoint,
                                    a2a_agent_name=params.a2a_agent_name,
                                    turn_number=params.turn_number,
                                    is_multiturn=params.is_multiturn,
                                    from_task=params.from_task,
                                    from_agent=params.from_agent,
                                ),
                            )
                            chunk_index += 1

                # Handle artifact updates
                elif is_stream_artifact_update(chunk):
                    artifact_update = chunk.artifact_update
                    artifact = artifact_update.artifact
                    if artifact and artifact.parts:
                        result_parts.extend(
                            part_text(part)
                            for part in artifact.parts
                            if part_is_text(part)
                        )
                        artifact_size = None
                        if artifact.parts:
                            artifact_size = sum(
                                len(part_text(p).encode())
                                if part_is_text(p)
                                else len(getattr(p, "raw", b""))
                                for p in artifact.parts
                            )
                        effective_context_id = (
                            (current_task.context_id if current_task else None)
                            or params.context_id
                        )
                        crewai_event_bus.emit(
                            agent_branch,
                            A2AArtifactReceivedEvent(
                                task_id=artifact_update.task_id or current_task_id,
                                artifact_id=artifact.artifact_id,
                                artifact_name=artifact.name,
                                artifact_description=artifact.description,
                                mime_type="text" if artifact.parts and part_is_text(artifact.parts[0]) else None,
                                size_bytes=artifact_size,
                                append=artifact_update.append or False,
                                last_chunk=artifact_update.last_chunk or False,
                                endpoint=params.endpoint,
                                a2a_agent_name=params.a2a_agent_name,
                                context_id=effective_context_id,
                                turn_number=params.turn_number,
                                is_multiturn=params.is_multiturn,
                                from_task=params.from_task,
                                from_agent=params.from_agent,
                            ),
                        )

                # Handle status updates
                elif is_stream_status_update(chunk):
                    update = chunk.status_update
                    is_final_update = process_status_update(update, result_parts)

                    if current_task and (
                        is_final_update
                        or current_task.status.state
                        in TERMINAL_STATES | ACTIONABLE_STATES
                    ):
                        final_result = process_task_state(
                            a2a_task=current_task,
                            new_messages=new_messages,
                            agent_card=agent_card,
                            turn_number=params.turn_number,
                            is_multiturn=params.is_multiturn,
                            agent_role=params.agent_role,
                            result_parts=result_parts,
                            endpoint=params.endpoint,
                            a2a_agent_name=params.a2a_agent_name,
                            from_task=params.from_task,
                            from_agent=params.from_agent,
                            is_final=is_final_update,
                        )
                    elif not current_task and is_final_update:
                        pass
                    else:
                        continue

        except A2AClientError as e:
            logger.warning(
                "Stream interrupted",
                extra={
                    "task_id": current_task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            if current_task_id:
                recovery_result = await StreamingHandler._try_recover_from_interruption(
                    client=client,
                    task_id=current_task_id,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    result_parts=result_parts,
                    **kwargs,
                )
                if recovery_result:
                    return recovery_result

            error_msg = f"A2A Client Error: {e!s}"
            error_message = new_text_message(
                error_msg,
                role=ROLE_AGENT,
                context_id=params.context_id,
                task_id=current_task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=str(e),
                    error_type="client_error",
                    a2a_agent_name=params.a2a_agent_name,
                    operation="streaming",
                    context_id=params.context_id,
                    task_id=current_task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AResponseReceivedEvent(
                    response=error_msg,
                    turn_number=params.turn_number,
                    context_id=params.context_id,
                    is_multiturn=params.is_multiturn,
                    status="failed",
                    final=True,
                    agent_role=params.agent_role,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TASK_STATE_FAILED,
                error=error_msg,
                history=new_messages,
            )

        except Exception as e:
            logger.warning(
                "Unexpected stream error",
                extra={
                    "task_id": current_task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            if current_task_id:
                recovery_result = await StreamingHandler._try_recover_from_interruption(
                    client=client,
                    task_id=current_task_id,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    result_parts=result_parts,
                    **kwargs,
                )
                if recovery_result:
                    return recovery_result

            error_msg = f"Unexpected error during streaming: {e!s}"
            error_message = new_text_message(
                error_msg,
                role=ROLE_AGENT,
                context_id=params.context_id,
                task_id=current_task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=str(e),
                    error_type="unexpected_error",
                    a2a_agent_name=params.a2a_agent_name,
                    operation="streaming",
                    context_id=params.context_id,
                    task_id=current_task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AResponseReceivedEvent(
                    response=error_msg,
                    turn_number=params.turn_number,
                    context_id=params.context_id,
                    is_multiturn=params.is_multiturn,
                    status="failed",
                    final=True,
                    agent_role=params.agent_role,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TASK_STATE_FAILED,
                error=error_msg,
                history=new_messages,
            )

        if final_result:
            return final_result

        response_text = " ".join(result_parts) if result_parts else ""
        crewai_event_bus.emit(
            agent_branch,
            A2AResponseReceivedEvent(
                response=response_text,
                turn_number=params.turn_number,
                context_id=params.context_id,
                is_multiturn=params.is_multiturn,
                status="completed",
                final=True,
                agent_role=params.agent_role,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
                from_task=params.from_task,
                from_agent=params.from_agent,
            ),
        )
        return TaskStateResult(
            status=TASK_STATE_FAILED,
            error="Stream ended without terminal state",
            history=new_messages,
        )
