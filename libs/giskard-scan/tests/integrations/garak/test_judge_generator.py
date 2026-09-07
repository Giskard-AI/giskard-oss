import asyncio
from typing import Any

import pytest

pytest.importorskip("garak")

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAICompatible
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.llm.types import (
    AssistantMessage,
    Choice,
    CompletionResponse,
)
from giskard.scan.integrations.garak._judge_generator import (
    GiskardJudgeGenerator,
    _conversation_to_messages,
)


class _EchoGenerator(BaseGenerator):
    """Records the messages it receives and returns a fixed judge verdict."""

    model_config = {"extra": "allow"}

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Any] = []

    async def _complete(
        self,
        messages,
        params: GenerationParams | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.seen.append(messages)
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content="Rating: [[9]]"),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )

    async def _call_model(
        self,
        messages,
        params: GenerationParams | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        return await self._complete(messages, params, metadata)


def _conversation(system: str, user: str) -> Conversation:
    return Conversation(
        turns=[
            Turn(role="system", content=Message(text=system)),
            Turn(role="user", content=Message(text=user)),
        ]
    )


def test_is_openai_compatible_subclass():
    gen = GiskardJudgeGenerator(_EchoGenerator())
    # judge.ModelAsJudge enforces this isinstance() gate.
    assert isinstance(gen, OpenAICompatible)


def test_construction_builds_no_client_and_reads_no_key(monkeypatch):
    monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)
    gen = GiskardJudgeGenerator(_EchoGenerator(), name="judge-x")
    assert gen.client is None
    assert gen.name == "judge-x"
    # no-op: reloading must not build an OpenAI client
    gen._load_unsafe()
    assert gen.client is None


def test_conversation_to_messages_maps_roles():
    conv = _conversation("you are a judge", "was the goal achieved?")
    messages = _conversation_to_messages(conv)
    assert messages == [
        {"role": "system", "content": "you are a judge"},
        {"role": "user", "content": "was the goal achieved?"},
    ]


def test_call_model_bridges_to_giskard_generator():
    echo = _EchoGenerator()
    loop = asyncio.new_event_loop()
    try:
        gen = GiskardJudgeGenerator(echo, loop=loop)
        conv = _conversation("system", "prompt")

        async def _run():
            return await asyncio.to_thread(gen._call_model, conv, 1)

        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert result and all(m is not None for m in result)
    assert [m.text for m in result if m is not None] == ["Rating: [[9]]"]
    # the giskard generator saw the messages (as ChatMessage objects after validation)
    assert echo.seen and len(echo.seen[0]) == 2
    assert echo.seen[0][0].role == "system"
