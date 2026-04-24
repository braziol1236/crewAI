"""Tests for a2a-sdk v1.0 compatibility.

These tests validate that crewai.a2a modules correctly import and work with
a2a-sdk v1.0.x (protobuf-based types). They cover the core issue described
in https://github.com/crewAIInc/crewAI/issues/5607:

    ImportError: cannot import name 'A2AClientHTTPError' from 'a2a.client.errors'

The migration from a2a-sdk ~0.3.10 to >=1.0.0,<2 introduced major breaking
changes including protobuf-based types, renamed error classes, and new enum
value conventions.
"""

from __future__ import annotations

import uuid

import pytest


class TestSdkV1Imports:
    """Verify that old v0.3 names no longer exist, and our compat layer works."""

    def test_a2a_client_error_importable(self) -> None:
        """A2AClientError (renamed from A2AClientHTTPError) should be importable."""
        from a2a.client.errors import A2AClientError

        assert A2AClientError is not None

    def test_old_a2a_client_http_error_removed(self) -> None:
        """A2AClientHTTPError no longer exists in a2a-sdk v1.0."""
        with pytest.raises(ImportError):
            from a2a.client.errors import A2AClientHTTPError  # noqa: F401

    def test_compat_alias_maps_to_new_error(self) -> None:
        """Our _compat alias should map to the new error class."""
        from a2a.client.errors import A2AClientError

        from crewai.a2a._compat import A2AClientHTTPError

        assert A2AClientHTTPError is A2AClientError

    def test_text_part_removed_in_v1(self) -> None:
        """TextPart no longer exists as a separate type in a2a-sdk v1.0."""
        with pytest.raises(ImportError):
            from a2a.types import TextPart  # noqa: F401

    def test_protobuf_types_importable(self) -> None:
        """Key protobuf types should be importable from a2a.types."""
        from a2a.types import (  # noqa: F401
            AgentCapabilities,
            AgentCard,
            AgentInterface,
            GetTaskRequest,
            Message,
            Part,
            Role,
            StreamResponse,
            SubscribeToTaskRequest,
            Task,
            TaskPushNotificationConfig,
            TaskState,
            TaskStatusUpdateEvent,
        )


class TestCompatLayer:
    """Tests for the crewai.a2a._compat compatibility layer."""

    def test_role_constants(self) -> None:
        """ROLE_USER and ROLE_AGENT should be valid Role enum values."""
        from a2a.types import Role

        from crewai.a2a._compat import ROLE_AGENT, ROLE_USER

        assert ROLE_USER == Role.ROLE_USER
        assert ROLE_AGENT == Role.ROLE_AGENT

    def test_task_state_constants(self) -> None:
        """TASK_STATE_* should be valid TaskState enum values."""
        from a2a.types import TaskState

        from crewai.a2a._compat import (
            TASK_STATE_CANCELED,
            TASK_STATE_COMPLETED,
            TASK_STATE_FAILED,
            TASK_STATE_INPUT_REQUIRED,
            TASK_STATE_REJECTED,
            TASK_STATE_SUBMITTED,
            TASK_STATE_WORKING,
        )

        assert TASK_STATE_SUBMITTED == TaskState.TASK_STATE_SUBMITTED
        assert TASK_STATE_WORKING == TaskState.TASK_STATE_WORKING
        assert TASK_STATE_COMPLETED == TaskState.TASK_STATE_COMPLETED
        assert TASK_STATE_FAILED == TaskState.TASK_STATE_FAILED
        assert TASK_STATE_CANCELED == TaskState.TASK_STATE_CANCELED
        assert TASK_STATE_INPUT_REQUIRED == TaskState.TASK_STATE_INPUT_REQUIRED
        assert TASK_STATE_REJECTED == TaskState.TASK_STATE_REJECTED

    def test_terminal_states(self) -> None:
        """TERMINAL_STATES should include completed, failed, rejected, canceled."""
        from crewai.a2a._compat import (
            TASK_STATE_CANCELED,
            TASK_STATE_COMPLETED,
            TASK_STATE_FAILED,
            TASK_STATE_REJECTED,
            TERMINAL_STATES,
        )

        assert TASK_STATE_COMPLETED in TERMINAL_STATES
        assert TASK_STATE_FAILED in TERMINAL_STATES
        assert TASK_STATE_REJECTED in TERMINAL_STATES
        assert TASK_STATE_CANCELED in TERMINAL_STATES


class TestPartHelpers:
    """Tests for protobuf Part helpers."""

    def test_new_text_part(self) -> None:
        """new_text_part should create a Part with text field set."""
        from crewai.a2a._compat import new_text_part, part_is_text, part_text

        part = new_text_part("hello world")
        assert part_is_text(part)
        assert part_text(part) == "hello world"

    def test_part_is_text_false_for_non_text(self) -> None:
        """part_is_text should return False for non-text parts."""
        from a2a.types import Part
        from google.protobuf.struct_pb2 import Value

        from crewai.a2a._compat import part_is_text

        v = Value()
        v.string_value = "test"
        part = Part(data=v)
        assert not part_is_text(part)

    def test_part_has_data(self) -> None:
        """part_has_data should detect data parts."""
        from a2a.types import Part
        from google.protobuf.struct_pb2 import Value

        from crewai.a2a._compat import part_has_data

        v = Value()
        v.string_value = "test"
        part = Part(data=v)
        assert part_has_data(part)

    def test_part_has_file(self) -> None:
        """part_has_file should detect raw/url file parts."""
        from a2a.types import Part

        from crewai.a2a._compat import part_has_file

        raw_part = Part(raw=b"file content", media_type="application/pdf")
        assert part_has_file(raw_part)

        url_part = Part(url="https://example.com/file.pdf", media_type="application/pdf")
        assert part_has_file(url_part)


class TestMessageHelpers:
    """Tests for protobuf Message helpers."""

    def test_new_text_message(self) -> None:
        """new_text_message should create a Message with a text Part."""
        from crewai.a2a._compat import (
            ROLE_USER,
            new_text_message,
            part_is_text,
            part_text,
        )

        msg = new_text_message("test message", role=ROLE_USER)
        assert msg.role == ROLE_USER
        assert len(msg.parts) == 1
        assert part_is_text(msg.parts[0])
        assert part_text(msg.parts[0]) == "test message"

    def test_new_text_message_with_context_and_task(self) -> None:
        """new_text_message should accept context_id and task_id."""
        from crewai.a2a._compat import ROLE_AGENT, new_text_message

        msg = new_text_message(
            "response",
            role=ROLE_AGENT,
            context_id="ctx-123",
            task_id="task-456",
        )
        assert msg.context_id == "ctx-123"
        assert msg.task_id == "task-456"


class TestAgentCardHelpers:
    """Tests for protobuf AgentCard helpers."""

    def test_agent_card_to_dict(self) -> None:
        """agent_card_to_dict should serialize an AgentCard to a plain dict."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import agent_card_to_dict

        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            supported_interfaces=[
                AgentInterface(url="http://localhost:9999", protocol_binding="JSONRPC"),
            ],
            version="1.0.0",
        )
        result = agent_card_to_dict(card)
        assert isinstance(result, dict)
        assert result["name"] == "Test Agent"
        assert result["description"] == "A test agent"

    def test_agent_card_url(self) -> None:
        """agent_card_url should return the URL from the first interface."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import agent_card_url

        card = AgentCard(
            name="Test",
            supported_interfaces=[
                AgentInterface(url="http://localhost:9999", protocol_binding="JSONRPC"),
            ],
        )
        assert agent_card_url(card) == "http://localhost:9999"

    def test_agent_card_url_empty_when_no_interfaces(self) -> None:
        """agent_card_url should return empty string if no interfaces."""
        from a2a.types import AgentCard

        from crewai.a2a._compat import agent_card_url

        card = AgentCard(name="No Interfaces")
        assert agent_card_url(card) == ""

    def test_agent_card_preferred_transport(self) -> None:
        """agent_card_preferred_transport should return protocol_binding."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import agent_card_preferred_transport

        card = AgentCard(
            name="Test",
            supported_interfaces=[
                AgentInterface(url="http://localhost", protocol_binding="GRPC"),
            ],
        )
        assert agent_card_preferred_transport(card) == "GRPC"

    def test_agent_card_interfaces(self) -> None:
        """agent_card_interfaces should return all interfaces."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import agent_card_interfaces

        card = AgentCard(
            name="Test",
            supported_interfaces=[
                AgentInterface(url="http://a.com", protocol_binding="JSONRPC"),
                AgentInterface(url="http://b.com", protocol_binding="GRPC"),
            ],
        )
        interfaces = agent_card_interfaces(card)
        assert len(interfaces) == 2

    def test_agent_card_protocol_version(self) -> None:
        """agent_card_protocol_version should return protocol version from first interface."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import agent_card_protocol_version

        card = AgentCard(
            name="Test",
            supported_interfaces=[
                AgentInterface(
                    url="http://localhost",
                    protocol_binding="JSONRPC",
                    protocol_version="0.3",
                ),
            ],
        )
        assert agent_card_protocol_version(card) == "0.3"


class TestProtoCopy:
    """Tests for protobuf deep copy helper."""

    def test_proto_copy_creates_independent_copy(self) -> None:
        """proto_copy should create a deep copy of a protobuf message."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import proto_copy

        original = AgentCard(
            name="Original",
            supported_interfaces=[
                AgentInterface(url="http://original.com", protocol_binding="JSONRPC"),
            ],
        )
        copy = proto_copy(original)
        copy.name = "Modified"

        assert original.name == "Original"
        assert copy.name == "Modified"


class TestStreamResponseHelpers:
    """Tests for StreamResponse event helpers."""

    def test_is_stream_message(self) -> None:
        """is_stream_message should detect messages in StreamResponse."""
        from a2a.types import Message, StreamResponse

        from crewai.a2a._compat import ROLE_AGENT, is_stream_message, new_text_part

        msg = Message(
            role=ROLE_AGENT,
            parts=[new_text_part("hello")],
            message_id=str(uuid.uuid4()),
        )
        sr = StreamResponse(message=msg)
        assert is_stream_message(sr)

    def test_is_stream_task(self) -> None:
        """is_stream_task should detect tasks in StreamResponse."""
        from a2a.types import StreamResponse, Task, TaskState, TaskStatus

        from crewai.a2a._compat import is_stream_task

        task = Task(
            id="task-1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )
        sr = StreamResponse(task=task)
        assert is_stream_task(sr)

    def test_is_stream_status_update(self) -> None:
        """is_stream_status_update should detect status updates."""
        from a2a.types import StreamResponse, TaskState, TaskStatus, TaskStatusUpdateEvent

        from crewai.a2a._compat import is_stream_status_update

        update = TaskStatusUpdateEvent(
            task_id="task-1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )
        sr = StreamResponse(status_update=update)
        assert is_stream_status_update(sr)


class TestStatusUpdateFinality:
    """Tests for status update finality detection."""

    def test_completed_is_final(self) -> None:
        """Completed status should be final."""
        from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

        from crewai.a2a._compat import is_status_update_final

        update = TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )
        assert is_status_update_final(update) is True

    def test_working_is_not_final(self) -> None:
        """Working status should not be final."""
        from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

        from crewai.a2a._compat import is_status_update_final

        update = TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )
        assert is_status_update_final(update) is False

    def test_failed_is_final(self) -> None:
        """Failed status should be final."""
        from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

        from crewai.a2a._compat import is_status_update_final

        update = TaskStatusUpdateEvent(
            task_id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
        )
        assert is_status_update_final(update) is True


class TestClientConfigHelper:
    """Tests for client configuration helper."""

    def test_create_client_config(self) -> None:
        """create_client_config should produce a valid ClientConfig."""
        from crewai.a2a._compat import create_client_config

        config = create_client_config(
            supported_transports=["JSONRPC", "GRPC"],
            streaming=True,
            polling=False,
        )
        assert config.supported_protocol_bindings == ["JSONRPC", "GRPC"]
        assert config.streaming is True
        assert config.polling is False


class TestProtoToJson:
    """Tests for proto_to_json serialization."""

    def test_proto_to_json(self) -> None:
        """proto_to_json should serialize a protobuf to JSON string."""
        from a2a.types import AgentCard, AgentInterface

        from crewai.a2a._compat import proto_to_json

        card = AgentCard(
            name="Test Agent",
            supported_interfaces=[
                AgentInterface(url="http://localhost:9999", protocol_binding="JSONRPC"),
            ],
        )
        json_str = proto_to_json(card)
        assert isinstance(json_str, str)
        assert "Test Agent" in json_str


class TestModuleImports:
    """Verify all crewai.a2a submodules import without error under v1.0."""

    def test_import_compat(self) -> None:
        from crewai.a2a._compat import A2AClientHTTPError  # noqa: F401

    def test_import_task_helpers(self) -> None:
        from crewai.a2a.task_helpers import process_task_state  # noqa: F401

    def test_import_polling_handler(self) -> None:
        from crewai.a2a.updates.polling.handler import PollingHandler  # noqa: F401

    def test_import_streaming_handler(self) -> None:
        from crewai.a2a.updates.streaming.handler import StreamingHandler  # noqa: F401

    def test_import_push_handler(self) -> None:
        from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler  # noqa: F401

    def test_import_auth_utils(self) -> None:
        from crewai.a2a.auth.utils import validate_auth_against_agent_card  # noqa: F401

    def test_import_delegation(self) -> None:
        from crewai.a2a.utils.delegation import execute_a2a_delegation  # noqa: F401

    def test_import_transport(self) -> None:
        from crewai.a2a.utils.transport import negotiate_transport  # noqa: F401

    def test_import_agent_card(self) -> None:
        from crewai.a2a.utils.agent_card import afetch_agent_card  # noqa: F401

    def test_import_agent_card_signing(self) -> None:
        from crewai.a2a.utils.agent_card_signing import sign_agent_card  # noqa: F401

    def test_import_wrapper(self) -> None:
        from crewai.a2a.wrapper import wrap_agent_with_a2a_instance  # noqa: F401

    def test_import_extensions_registry(self) -> None:
        from crewai.a2a.extensions.registry import ExtensionsMiddleware  # noqa: F401

    def test_import_content_type(self) -> None:
        from crewai.a2a.utils.content_type import get_part_content_type  # noqa: F401


class TestGetPartContentType:
    """Tests for get_part_content_type with v1.0 protobuf Parts."""

    def test_text_part_returns_text_plain(self) -> None:
        from a2a.types import Part

        from crewai.a2a.utils.content_type import get_part_content_type

        part = Part(text="hello")
        assert get_part_content_type(part) == "text/plain"

    def test_data_part_returns_application_json(self) -> None:
        from a2a.types import Part
        from google.protobuf.struct_pb2 import Value

        from crewai.a2a.utils.content_type import get_part_content_type

        v = Value()
        v.string_value = "test"
        part = Part(data=v)
        assert get_part_content_type(part) == "application/json"

    def test_raw_part_returns_media_type(self) -> None:
        from a2a.types import Part

        from crewai.a2a.utils.content_type import get_part_content_type

        part = Part(raw=b"pdf content", media_type="application/pdf")
        assert get_part_content_type(part) == "application/pdf"

    def test_url_part_returns_media_type(self) -> None:
        from a2a.types import Part

        from crewai.a2a.utils.content_type import get_part_content_type

        part = Part(url="https://example.com/image.png", media_type="image/png")
        assert get_part_content_type(part) == "image/png"
