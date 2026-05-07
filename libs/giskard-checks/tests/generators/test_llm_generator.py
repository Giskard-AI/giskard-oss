# libs/giskard-checks/tests/generators/test_llm_generator.py
import json
from collections.abc import Sequence
from typing import Any, override

import pytest
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Interaction, Trace
from giskard.checks.generators.base import LLMGenerator
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field


class MockGenerator(BaseGenerator):
    responses: list[dict[str, Any]]
    index: int = 0
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        message = AssistantMessage(content=json.dumps(self.responses[self.index]))
        self.index += 1
        return CompletionResponse(
            choices=[Choice(message=message, finish_reason="stop", index=0)]
        )


class LLMTrace(Trace[str, str], frozen=True):
    def _repr_prompt_(self) -> str:
        if not self.interactions:
            return "**No interactions yet**"
        return "\n".join(
            f"[user]: {i.inputs}\n[assistant]: {i.outputs}" for i in self.interactions
        )


def test_llm_generator_requires_prompt_or_prompt_path():
    with pytest.raises(ValueError, match="prompt.*prompt_path"):
        _ = LLMGenerator(prompt=None, prompt_path=None)


def test_llm_generator_rejects_both_prompt_and_prompt_path():
    with pytest.raises(ValueError, match="both"):
        _ = LLMGenerator(prompt="hello", prompt_path="some::path.j2")


def test_llm_generator_accepts_prompt():
    gen = LLMGenerator(prompt="You are a user. Say hello.")
    assert gen.prompt == "You are a user. Say hello."
    assert gen.prompt_path is None


def test_llm_generator_accepts_prompt_path():
    gen = LLMGenerator(prompt_path="giskard.checks::generators/user_simulator.j2")
    assert gen.prompt_path == "giskard.checks::generators/user_simulator.j2"
    assert gen.prompt is None


def test_llm_generator_default_max_steps():
    gen = LLMGenerator(prompt="hello")
    assert gen.max_steps == 3


def test_llm_generator_registered_as_kind():
    from giskard.checks.core.input_generator import InputGenerator

    gen = InputGenerator.model_validate({"kind": "llm_generator", "prompt": "hello"})
    assert isinstance(gen, LLMGenerator)


@pytest.mark.asyncio
async def test_llm_generator_yields_message_and_stops_on_goal_reached():
    mock_gen = MockGenerator(
        responses=[
            {"goal_reached": False, "message": "Hello there"},
            {"goal_reached": True, "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say hello.", max_steps=5)
    trace = LLMTrace()
    agen = gen(trace)
    msg = await anext(agen)
    assert msg == "Hello there"

    trace = await trace.with_interaction(Interaction(inputs=msg, outputs="Hi!"))
    with pytest.raises(StopAsyncIteration):
        _ = await agen.asend(trace)

    assert len(mock_gen.calls) == 2


@pytest.mark.asyncio
async def test_llm_generator_stops_at_max_steps():
    mock_gen = MockGenerator(
        responses=[
            {"goal_reached": False, "message": "Step 1"},
            {"goal_reached": False, "message": "Step 2"},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Keep going.", max_steps=1)
    trace = LLMTrace()
    agen = gen(trace)
    msg = await anext(agen)
    assert msg == "Step 1"

    trace = await trace.with_interaction(Interaction(inputs=msg, outputs="ok"))
    with pytest.raises(StopAsyncIteration):
        _ = await agen.asend(trace)

    assert len(mock_gen.calls) == 1
