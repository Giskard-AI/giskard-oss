import json
from typing import Any, override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import Interaction, PersonaSimulator, Trace
from pydantic import Field


class MockPersonaGenerator(BaseGenerator):
    """Mock generator for PersonaSimulator tests."""

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


def create_mock_response(
    goal_reached: bool,
    message: str | None,
    client_description: str | None = None,
) -> dict[str, Any]:
    """Helper to create mock response dictionaries."""
    return {
        "client_description": client_description,
        "goal_reached": goal_reached,
        "message": message,
    }


async def advance_turn(
    gen, trace: LLMTrace, response_text: str
) -> tuple[LLMTrace, str]:
    """Helper to advance generator by one turn and return updated trace and next input."""
    next_input = await gen.asend(trace)
    updated_trace = await trace.with_interaction(
        Interaction(inputs=next_input, outputs=response_text)
    )
    return updated_trace, next_input


@pytest.mark.parametrize(
    "persona,context,description",
    [
        ("frustrated_customer", None, "predefined persona without context"),
        ("frustrated_customer", "delayed order", "predefined persona with context"),
        (
            "A polite elderly user who needs step-by-step guidance",
            None,
            "custom persona without context",
        ),
        (
            "A busy executive",
            "Looking for quick answers",
            "custom persona with context",
        ),
    ],
)
def test_persona_and_context_assignment(persona, context, description):
    """Test persona and context field assignments."""
    simulator = PersonaSimulator(persona=persona, context=context)
    assert simulator.persona == persona
    assert simulator.context == context


def test_empty_persona_rejected():
    """Test that empty persona string is rejected."""
    with pytest.raises(ValueError, match="at least 1 character"):
        PersonaSimulator(persona="")


def test_negative_max_steps_rejected():
    """Test that negative max_steps is rejected."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        PersonaSimulator(persona="test_user", max_steps=-1)


async def test_persona_simulator_first_turn_generates_client_description():
    """Test that first turn generates and stores client description."""
    client_desc = "Sarah Martinez, 35, working professional"
    generator = MockPersonaGenerator(
        responses=[
            create_mock_response(False, "Hi, I need help with my order", client_desc),
            create_mock_response(True, None),
        ]
    )

    simulator = PersonaSimulator(
        generator=generator,
        persona="frustrated_customer",
        context="Order delayed 5 days",
        max_steps=2,
    )

    trace = LLMTrace()
    gen = simulator(trace)

    first_input = await anext(gen)
    assert first_input == "Hi, I need help with my order"
    assert simulator._client_description == client_desc
    first_call_content = str(generator.calls[0][-1].content)
    assert "frustrated" in first_call_content.lower()

    trace = await trace.with_interaction(
        Interaction(inputs=first_input, outputs="How can I help?")
    )
    with pytest.raises(StopAsyncIteration):
        await gen.asend(trace)

    second_call_content = str(generator.calls[1][-1].content)
    assert "Sarah Martinez" in second_call_content


async def test_persona_simulator_maintains_client_description_across_turns():
    """Test that same client description is used for all turns."""
    client_desc = "John Doe, 40, tech-savvy"
    generator = MockPersonaGenerator(
        responses=[
            create_mock_response(False, "First message", client_desc),
            create_mock_response(False, "Second message"),
            create_mock_response(True, None),
        ]
    )

    simulator = PersonaSimulator(generator=generator, persona="power_user", max_steps=3)

    trace = LLMTrace()
    gen = simulator(trace)

    input1 = await anext(gen)
    assert input1 == "First message"
    assert simulator._client_description == client_desc

    trace, input2 = await advance_turn(gen, trace, "Response 1")
    assert input2 == "Second message"
    assert simulator._client_description == client_desc

    trace = await trace.with_interaction(
        Interaction(inputs=input2, outputs="Response 2")
    )
    with pytest.raises(StopAsyncIteration):
        await gen.asend(trace)

    assert simulator._client_description == client_desc


async def test_persona_simulator_respects_max_steps():
    """Test that simulator respects max_steps limit."""
    generator = MockPersonaGenerator(
        responses=[
            create_mock_response(False, "Message 1", "Test User"),
            create_mock_response(False, "Message 2"),
        ]
    )

    simulator = PersonaSimulator(
        generator=generator, persona="helpful_user", max_steps=1
    )

    trace = LLMTrace()
    gen = simulator(trace)

    first_input = await anext(gen)
    assert first_input == "Message 1"
    assert len(generator.calls) == 1

    trace = await trace.with_interaction(
        Interaction(inputs=first_input, outputs="Response")
    )
    with pytest.raises(StopAsyncIteration):
        await gen.asend(trace)

    assert len(generator.calls) == 1
