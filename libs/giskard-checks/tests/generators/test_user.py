import json
from typing import Any, override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import Interaction, Trace, UserSimulator
from pydantic import Field


class MockGenerator(BaseGenerator):
    """Mock generator for UserSimulator tests."""

    responses: list[dict[str, Any]]
    index: int = 0
    calls: list[list[Message]] = Field(default_factory=list)

    @override
    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        self.calls.append(messages)
        response = Response(
            message=Message(
                role="assistant",
                content=json.dumps(self.responses[self.index]),
            ),
            finish_reason="stop",
        )
        self.index += 1
        return response


class LLMTrace(Trace[str, str], frozen=True):
    def _repr_prompt_(self) -> str:
        if not self.interactions:
            return "**No interactions yet**"
        return "\n".join(
            [
                f"[user]: {interaction.inputs}\n[assistant]: {interaction.outputs}"
                for interaction in self.interactions
            ]
        )


def _wrap_in_xml_tag(text: str, tag: str) -> str:
    return f"<{tag}>\n{text}\n</{tag}>"


def create_mock_response(
    goal_reached: bool,
    message: str | None,
) -> dict[str, Any]:
    """Helper to create mock response dictionaries."""
    return {
        "goal_reached": goal_reached,
        "message": message,
    }


@pytest.mark.parametrize(
    "persona,context",
    [
        ("frustrated_customer", None),
        ("frustrated_customer", "delayed order"),
        ("A polite elderly user who needs step-by-step guidance", None),
        ("A busy executive", "Looking for quick answers"),
    ],
    ids=[
        "persona without context",
        "persona with context",
        "custom persona without context",
        "custom persona with context",
    ],
)
def test_persona_and_context_assignment(persona, context):
    """Test persona and context field assignments."""
    simulator = UserSimulator(persona=persona, context=context)
    assert simulator.persona == persona
    assert simulator.context == context


def test_empty_persona_rejected():
    """Test that empty persona string is rejected."""
    with pytest.raises(ValueError, match="at least 1 character"):
        UserSimulator(persona="")


def test_negative_max_steps_rejected():
    """Test that negative max_steps is rejected."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        UserSimulator(persona="test_user", max_steps=-1)


async def test_user_simulator_returns_messages_until_goal_reached():
    generator = MockGenerator(
        responses=[
            create_mock_response(False, "Hello, how are you?"),
            create_mock_response(True, None),
        ]
    )
    user_simulator = UserSimulator(
        generator=generator, persona="Greet the chatbot", max_steps=2
    )

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[0][-1].content
    )

    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="I'm good, thank you!")
    )
    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    assert len(generator.calls) == 2
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[1][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[1][-1].content
    )


async def test_user_simulator_returns_messages_until_max_steps():
    generator = MockGenerator(
        responses=[
            create_mock_response(False, "Hello, how are you?"),
            create_mock_response(False, "I'm good too"),
            create_mock_response(True, None),
        ]
    )
    user_simulator = UserSimulator(
        generator=generator, persona="Greet the chatbot", max_steps=1
    )

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert len(generator.calls) == 1
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[0][-1].content
    )

    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="I'm good and you?")
    )
    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    assert len(generator.calls) == 1


async def test_user_simulator_multiple_steps():
    generator = MockGenerator(
        responses=[
            create_mock_response(False, "Hello, how are you?"),
            create_mock_response(False, "I'm good too"),
            create_mock_response(True, None),
        ]
    )
    user_simulator = UserSimulator(generator=generator, persona="Greet the chatbot")

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert len(generator.calls) == 1
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[0][-1].content
    )

    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="I'm good and you?")
    )
    inputs = await gen.asend(trace)
    assert inputs == "I'm good too"

    assert len(generator.calls) == 2
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[1][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[1][-1].content
    )

    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="How do I get to the city center?")
    )
    with pytest.raises(StopAsyncIteration):
        inputs = await gen.asend(trace)

    assert len(generator.calls) == 3
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[2][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.persona, "persona") in str(
        generator.calls[2][-1].content
    )
