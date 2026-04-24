"""Common parameter extraction for streaming handlers."""

from __future__ import annotations

from a2a.types import TaskStatusUpdateEvent

from crewai.a2a._compat import is_status_update_final, part_is_text, part_text


def process_status_update(
    update: TaskStatusUpdateEvent,
    result_parts: list[str],
) -> bool:
    """Process a status update event and extract text parts.

    Args:
        update: The status update event.
        result_parts: List to append text parts to (modified in place).

    Returns:
        True if this is a final update, False otherwise.
    """
    is_final = is_status_update_final(update)
    if update.status and update.status.message and update.status.message.parts:
        result_parts.extend(
            part_text(part)
            for part in update.status.message.parts
            if part_is_text(part) and part_text(part)
        )
    return is_final
