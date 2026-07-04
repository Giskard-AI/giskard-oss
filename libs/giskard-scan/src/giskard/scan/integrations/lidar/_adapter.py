"""Adapter that runs the private lidar scanner through Giskard scan."""

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

from giskard.checks import Interaction, Trace

if TYPE_CHECKING:
    # Type-only: lidar's compat Message union. Never imported at runtime — the
    # function only reads .role/.content, so no lidar dependency is introduced.
    from lidar.giskard_compat import Message

logger = logging.getLogger(__name__)


def lidar_available() -> bool:
    """Return True if the private lidar dependency is importable."""
    return find_spec("lidar") is not None


def _require_lidar() -> None:
    if not lidar_available():
        raise ImportError(
            "lidar is not installed. It is a private Giskard package and is not "
            "available on PyPI. Install it from git: "
            "pip install git+https://github.com/Giskard-AI/lidar.git@v0.2.7"
        )


async def _trace_from_messages(messages: "list[Message]") -> Trace:
    """Rebuild a scan Trace from a lidar attempt's flat message list.

    Lidar owns its own executor and hands back finished conversations, so we
    reconstruct a display trace by pairing each user turn with the assistant
    reply that follows it. System/tool messages carry no input/output pair and
    are skipped. A trailing user message with no reply yields outputs=None.
    """
    interactions: list[Interaction] = []
    pending_input: str | None = None
    for message in messages:
        if message.role == "user":
            if pending_input is not None:
                interactions.append(Interaction(inputs=pending_input, outputs=None))
            pending_input = message.content
        elif message.role == "assistant":
            if pending_input is not None:
                interactions.append(
                    Interaction(inputs=pending_input, outputs=message.content)
                )
                pending_input = None
        # system / tool messages: no input/output pairing, skip
    if pending_input is not None:
        interactions.append(Interaction(inputs=pending_input, outputs=None))
    return await Trace.from_interactions(*interactions)


class LidarScanAdapter:
    """Build and run a Giskard suite from a lidar scan. Filled in by later tasks."""
