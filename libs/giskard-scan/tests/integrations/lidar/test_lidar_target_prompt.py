"""The lidar bridge sends the target only the text of the latest message.

Runs without lidar: the two lidar modules ``_call_model`` imports lazily are
stubbed, so this checks the bridge's own message handling, not lidar.
"""

import sys
from types import ModuleType
from typing import Any

import pytest
from giskard.llm.types import (
    AssistantMessage,
    ReasoningContent,
    ReasoningText,
    TextContent,
    UserMessage,
)
from giskard.scan.integrations.lidar._target import ScanTargetGenerator


@pytest.fixture
def stub_lidar(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub lidar's call-id side channel and response factory."""
    responses: list[dict[str, Any]] = []

    target_module = ModuleType("lidar.core.models.target")
    target_module.get_current_target_call_id = lambda: None  # pyright: ignore[reportAttributeAccessIssue]
    compat_module = ModuleType("lidar.giskard_compat")
    compat_module.make_response = lambda **kwargs: responses.append(kwargs)  # pyright: ignore[reportAttributeAccessIssue]

    for name in ("lidar", "lidar.core", "lidar.core.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "lidar.core.models.target", target_module)
    monkeypatch.setitem(sys.modules, "lidar.giskard_compat", compat_module)
    return responses


def _echo(inputs: str) -> str:
    return f"echo: {inputs}"


async def test_bridge_prompts_target_with_text_parts_of_last_message(stub_lidar):
    bridge = ScanTargetGenerator(target=_echo)

    await bridge.complete(
        [UserMessage(content=[TextContent(text="hello"), TextContent(text="there")])]
    )

    assert stub_lidar[0]["content"] == "echo: hello\nthere"


async def test_bridge_never_prompts_target_with_reasoning(stub_lidar):
    bridge = ScanTargetGenerator(target=_echo)

    await bridge.complete(
        [
            AssistantMessage(
                content=[
                    ReasoningContent(
                        content=[ReasoningText(text="secret thought")],
                        provider="anthropic",
                    ),
                    TextContent(text="visible"),
                ]
            )
        ]
    )

    assert stub_lidar[0]["content"] == "echo: visible"
