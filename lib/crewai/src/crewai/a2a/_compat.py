"""Compatibility layer for a2a-sdk v0.3 → v1.0 migration.

Centralizes import aliases and helper functions so the rest of the
a2a module can use a single import regardless of SDK version.
"""

from __future__ import annotations

from typing import Any

from a2a.client.errors import A2AClientError


# ---------------------------------------------------------------------------
# Error re-exports
# In v0.3 the class was called A2AClientHTTPError; v1.0 renamed it to
# A2AClientError.  We expose the new name *and* an alias used across the
# codebase so callers can migrate incrementally.
# ---------------------------------------------------------------------------
A2AClientHTTPError = A2AClientError  # back-compat alias

# ---------------------------------------------------------------------------
# Type helpers - Protobuf Part access
# In v0.3 Part was a Pydantic discriminated-union with ``part.root.kind``
# and ``part.root.text``; in v1.0 Part is a protobuf message with a
# ``content`` oneof.
# ---------------------------------------------------------------------------

from a2a.types import Part  # noqa: E402


def part_is_text(part: Part) -> bool:
    """Return True when the Part carries text content."""
    return part.HasField("text")  # type: ignore[no-any-return]


def part_text(part: Part) -> str:
    """Return the text payload of a Part (assumes text content)."""
    return part.text


def part_has_data(part: Part) -> bool:
    """Return True when the Part carries structured data."""
    return part.HasField("data")  # type: ignore[no-any-return]


def part_has_file(part: Part) -> bool:
    """Return True when the Part carries a file (url or raw bytes)."""
    return part.HasField("url") or part.HasField("raw")  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Enum value aliases
# v0.3: TaskState.completed, Role.user  (lower snake_case strings)
# v1.0: TaskState.TASK_STATE_COMPLETED, Role.ROLE_USER  (SCREAMING_SNAKE_CASE)
# ---------------------------------------------------------------------------

from a2a.types import Role, TaskState  # noqa: E402


# TaskState aliases
TASK_STATE_SUBMITTED = TaskState.TASK_STATE_SUBMITTED
TASK_STATE_WORKING = TaskState.TASK_STATE_WORKING
TASK_STATE_COMPLETED = TaskState.TASK_STATE_COMPLETED
TASK_STATE_FAILED = TaskState.TASK_STATE_FAILED
TASK_STATE_CANCELED = TaskState.TASK_STATE_CANCELED
TASK_STATE_INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
TASK_STATE_AUTH_REQUIRED = TaskState.TASK_STATE_AUTH_REQUIRED
TASK_STATE_REJECTED = TaskState.TASK_STATE_REJECTED

# Role aliases
ROLE_USER = Role.ROLE_USER
ROLE_AGENT = Role.ROLE_AGENT


# ---------------------------------------------------------------------------
# Protobuf object helpers
# Protobuf objects don't have model_dump() / model_copy().
# ---------------------------------------------------------------------------

from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]  # noqa: E402


def proto_to_json(msg: Any) -> str:
    """Serialize a protobuf message to a JSON string.

    Replaces ``msg.model_dump_json(...)`` from v0.3 Pydantic models.
    """
    from google.protobuf.json_format import MessageToJson

    return MessageToJson(msg, preserving_proto_field_name=True, indent=2)  # type: ignore[no-any-return]


def agent_card_to_dict(agent_card: Any, *, exclude_none: bool = True) -> dict[str, Any]:
    """Serialize a protobuf AgentCard to a plain dict.

    Works like ``agent_card.model_dump(exclude_none=True)`` did in v0.3.
    """
    return MessageToDict(  # type: ignore[no-any-return]
        agent_card,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=not exclude_none,
    )


def proto_copy(msg: Any) -> Any:
    """Return a deep copy of a protobuf message (replaces model_copy)."""
    new = type(msg)()
    new.CopyFrom(msg)
    return new


# ---------------------------------------------------------------------------
# Message / Part construction helpers
# v0.3: Message(role=Role.user, parts=[Part(root=TextPart(text=...))])
# v1.0: Message(role=Role.ROLE_USER, parts=[Part(text=...)])
# ---------------------------------------------------------------------------

from a2a.types import Message  # noqa: E402


def new_text_part(text: str, **kwargs: Any) -> Part:
    """Create a Part with text content (v1.0 style)."""
    return Part(text=text, **kwargs)


def new_text_message(
    text: str,
    *,
    role: Any = ROLE_AGENT,
    message_id: str | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    **kwargs: Any,
) -> Message:
    """Create a Message with a single text Part."""
    import uuid as _uuid

    return Message(
        role=role,
        message_id=message_id or str(_uuid.uuid4()),
        parts=[Part(text=text)],
        context_id=context_id or "",
        task_id=task_id or "",
        **kwargs,
    )


def make_send_request(message: Message) -> Any:
    """Wrap a Message in a SendMessageRequest (v1.0 API).

    In v0.3, ``client.send_message(message)`` accepted a bare ``Message``.
    In v1.0, it expects ``SendMessageRequest(message=message)``.
    """
    from a2a.types import SendMessageRequest

    return SendMessageRequest(message=message)


# ---------------------------------------------------------------------------
# AgentCard field access helpers
# v0.3: agent_card.url, agent_card.preferred_transport, agent_card.additional_interfaces
# v1.0: agent_card.supported_interfaces, interface.url, interface.protocol_binding
# ---------------------------------------------------------------------------

from a2a.types import AgentCard, AgentInterface  # noqa: E402


def agent_card_url(agent_card: AgentCard) -> str:
    """Get the primary URL from an AgentCard.

    In v0.3 this was ``agent_card.url``.
    In v1.0 the URL lives inside ``supported_interfaces``.
    """
    if agent_card.supported_interfaces:
        return agent_card.supported_interfaces[0].url  # type: ignore[no-any-return]
    return ""


def agent_card_preferred_transport(agent_card: AgentCard) -> str:
    """Get the preferred transport protocol from an AgentCard.

    In v0.3 this was ``agent_card.preferred_transport``.
    In v1.0 it's the protocol_binding of the first supported_interface.
    """
    if agent_card.supported_interfaces:
        return agent_card.supported_interfaces[0].protocol_binding  # type: ignore[no-any-return]
    return "JSONRPC"


def agent_card_interfaces(agent_card: AgentCard) -> list[AgentInterface]:
    """Get all interfaces from an AgentCard.

    In v0.3 these were split between the primary url and
    ``agent_card.additional_interfaces``.
    In v1.0 everything is in ``supported_interfaces``.
    """
    return (
        list(agent_card.supported_interfaces) if agent_card.supported_interfaces else []
    )


def agent_card_protocol_version(agent_card: AgentCard) -> str:
    """Get the protocol version from an AgentCard.

    In v0.3 this was ``agent_card.protocol_version``.
    In v1.0 it's per-interface in ``interface.protocol_version``.
    """
    if agent_card.supported_interfaces:
        return agent_card.supported_interfaces[0].protocol_version or ""
    return ""


# ---------------------------------------------------------------------------
# StreamResponse helpers
# v0.3: send_message returned AsyncIterator[tuple[Task, Update] | Message]
# v1.0: send_message returns AsyncIterator[StreamResponse]
# ---------------------------------------------------------------------------
from a2a.types import (  # noqa: E402
    StreamResponse,
    TaskStatusUpdateEvent,
)


def is_stream_message(chunk: StreamResponse) -> bool:
    """Check if a StreamResponse contains a Message."""
    return chunk.HasField("message")  # type: ignore[no-any-return]


def is_stream_task(chunk: StreamResponse) -> bool:
    """Check if a StreamResponse contains a Task."""
    return chunk.HasField("task")  # type: ignore[no-any-return]


def is_stream_status_update(chunk: StreamResponse) -> bool:
    """Check if a StreamResponse contains a TaskStatusUpdateEvent."""
    return chunk.HasField("status_update")  # type: ignore[no-any-return]


def is_stream_artifact_update(chunk: StreamResponse) -> bool:
    """Check if a StreamResponse contains a TaskArtifactUpdateEvent."""
    return chunk.HasField("artifact_update")  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Client configuration helpers
# v0.3: ClientConfig.supported_transports, push_notification_configs (list)
# v1.0: ClientConfig.supported_protocol_bindings, push_notification_config (singular)
# ---------------------------------------------------------------------------

from a2a.client import ClientConfig  # noqa: E402
from a2a.types import TaskPushNotificationConfig  # noqa: E402


def create_client_config(
    *,
    httpx_client: Any = None,
    supported_transports: list[str] | None = None,
    streaming: bool = True,
    polling: bool = False,
    accepted_output_modes: list[str] | None = None,
    push_notification_config: TaskPushNotificationConfig | None = None,
    grpc_channel_factory: Any = None,
) -> ClientConfig:
    """Create a ClientConfig compatible with a2a-sdk v1.0."""
    return ClientConfig(
        httpx_client=httpx_client,
        supported_protocol_bindings=supported_transports or ["JSONRPC"],
        streaming=streaming,
        polling=polling,
        accepted_output_modes=accepted_output_modes
        or ["text/plain", "application/json"],
        push_notification_config=push_notification_config,
        grpc_channel_factory=grpc_channel_factory,
    )


# ---------------------------------------------------------------------------
# GetTaskRequest / SubscribeToTaskRequest
# v0.3: TaskQueryParams, TaskIdParams
# v1.0: GetTaskRequest, SubscribeToTaskRequest
# ---------------------------------------------------------------------------

from a2a.types import GetTaskRequest, SubscribeToTaskRequest  # noqa: E402


# Expose v0.3 names as aliases for the v1.0 types
TaskQueryParams = GetTaskRequest
TaskIdParams = SubscribeToTaskRequest


# ---------------------------------------------------------------------------
# Task status helpers
# v1.0 TaskStatusUpdateEvent no longer has a `final` field.  Finality is
# determined by the task state being terminal.
# ---------------------------------------------------------------------------

TERMINAL_STATES: frozenset[int] = frozenset(
    {
        TASK_STATE_COMPLETED,
        TASK_STATE_FAILED,
        TASK_STATE_REJECTED,
        TASK_STATE_CANCELED,
    }
)


def is_status_update_final(update: TaskStatusUpdateEvent) -> bool:
    """Determine if a status update is final.

    In v0.3 this was ``update.final``.  In v1.0 finality is inferred from
    the task state being terminal.
    """
    if update.status and update.status.state:
        return update.status.state in TERMINAL_STATES
    return False
