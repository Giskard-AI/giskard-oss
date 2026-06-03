"""Functional tests verifying that both generator backends work end-to-end.

Three scenarios are covered via pytest marks and auto-skip logic:

- ``@pytest.mark.litellm`` — runs when the optional ``litellm`` extra is
  installed. Exercises ``LiteLLMGenerator``.
- ``@pytest.mark.google`` — runs when the giskard-llm ``google`` extra is
  installed. Exercises ``GiskardLLMGenerator``.
- When both SDKs are installed both test variants run, proving that the
  two backends coexist in the same Python process.

All tests hit a live model and therefore require the relevant API key. They
are marked ``functional`` so ``make test-unit`` never picks them up.
"""

import os
from typing import Callable

import pytest
from giskard import agents
from giskard.agents.generators.base import BaseGenerator
from giskard.agents.generators.giskard_llm_generator import GiskardLLMGenerator
from giskard.llm.types import SystemMessage, UserMessage

pytestmark = pytest.mark.functional


def _giskard_llm_generator() -> GiskardLLMGenerator:
    return GiskardLLMGenerator(model=os.getenv("TEST_MODEL", "google/gemini-3.5-flash"))


def _litellm_generator() -> BaseGenerator:
    from giskard.agents.generators.litellm_generator import LiteLLMGenerator

    return LiteLLMGenerator(
        model=os.getenv("TEST_LITELLM_MODEL", "gemini/gemini-3.5-flash")
    )


# Each parametrize entry below is tagged with the provider mark that gates it
# via conftest auto-skip. That way the "litellm-only" and "giskard-llm-only"
# matrices skip the other backend while the "both" matrix runs both.
_BACKEND_CASES = [
    pytest.param(
        "giskard_llm",
        _giskard_llm_generator,
        marks=[pytest.mark.google, pytest.mark.giskard_llm],
        id="giskard-llm",
    ),
    pytest.param(
        "litellm",
        _litellm_generator,
        marks=pytest.mark.litellm,
        id="litellm",
    ),
]


@pytest.mark.parametrize("backend, generator_factory", _BACKEND_CASES)
async def test_backend_completion(
    backend: str, generator_factory: Callable[[], BaseGenerator]
) -> None:
    """Direct ``complete()`` call returns an assistant message."""
    generator = generator_factory()
    response = await generator.complete(
        messages=[
            SystemMessage(
                role="system",
                content="You are TestBot. Always include 'TestBot' in replies.",
            ),
            UserMessage(content="Say hi."),
        ]
    )

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.text is not None
    assert "testbot" in response.choices[0].message.text.lower()


@pytest.mark.parametrize("backend, generator_factory", _BACKEND_CASES)
async def test_backend_chat_workflow(
    backend: str, generator_factory: Callable[[], BaseGenerator]
) -> None:
    """``ChatWorkflow`` runs to completion on both backends."""
    workflow = agents.ChatWorkflow(generator=generator_factory())
    chat = await (
        workflow.chat("Your name is TestBot.", role="system")
        .chat("What is your name? Answer in one word.", role="user")
        .run()
    )

    assert chat.last.text is not None
    assert "testbot" in chat.last.text.lower()


@pytest.mark.litellm
@pytest.mark.google
async def test_backends_coexist_in_same_process() -> None:
    """Both generators can be instantiated and used in the same process.

    Only runs when both the ``litellm`` and ``google`` SDKs are installed —
    i.e. in the "both" CI matrix entry. Validates that the discriminator
    registry and the two adapters do not interfere with each other.
    """
    giskard_gen = _giskard_llm_generator()
    litellm_gen = _litellm_generator()

    giskard_response = await giskard_gen.complete(
        messages=[{"role": "user", "content": "Reply with just the digit 1."}]
    )
    litellm_response = await litellm_gen.complete(
        messages=[{"role": "user", "content": "Reply with just the digit 2."}]
    )

    assert giskard_response.choices[0].message.role == "assistant"
    assert litellm_response.choices[0].message.role == "assistant"
    assert giskard_response.choices[0].message.text is not None
    assert litellm_response.choices[0].message.text is not None
    assert "1" in giskard_response.choices[0].message.text
    assert "2" in litellm_response.choices[0].message.text
