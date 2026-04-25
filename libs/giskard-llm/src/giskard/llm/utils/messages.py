from collections.abc import Sequence

from ..types import ChatMessage


def extract_system_messages(messages: Sequence[ChatMessage]) -> list[str]:
    """Extract system and developer messages in order (both map to ``system``)."""
    return [m.content for m in messages if m.role == "system" or m.role == "developer"]
