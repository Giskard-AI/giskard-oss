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


def test_predefined_persona_accepted():
    """Test predefined persona is accepted."""
    simulator = PersonaSimulator(persona="frustrated_customer")
    assert simulator.persona == "frustrated_customer"


def test_predefined_persona_with_context():
    """Test predefined persona with additional context."""
    simulator = PersonaSimulator(persona="frustrated_customer", context="delayed order")
    assert simulator.persona == "frustrated_customer"
    assert simulator.context == "delayed order"


def test_custom_persona_description():
    """Test custom persona description (not predefined)."""
    custom = "A polite elderly user who needs step-by-step guidance"
    simulator = PersonaSimulator(persona=custom)
    assert simulator.persona == custom


def test_custom_persona_with_context():
    """Test custom persona with context."""
    custom = "A busy executive"
    context = "Looking for quick answers"
    simulator = PersonaSimulator(persona=custom, context=context)
    assert simulator.persona == custom
    assert simulator.context == context


async def test_persona_simulator_first_turn_generates_client_description():
    """Test that first turn generates and stores client description."""
    generator = MockPersonaGenerator(
        responses=[
            {
                "client_description": "Sarah Martinez, 35, working professional",
                "goal_reached": False,
                "message": "Hi, I need help with my order",
            },
            {
                "goal_reached": True,
                "message": None,
            },
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

    # First turn - should generate client description
    inputs = await anext(gen)
    assert inputs == "Hi, I need help with my order"
    assert simulator._client_description == "Sarah Martinez, 35, working professional"

    # Verify template received None for client_description (first turn)
    first_call_content = str(generator.calls[0][-1].content)
    assert "frustrated" in first_call_content.lower()

    # Second turn - should use stored client description
    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="How can I help?")
    )

    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    # Verify template received the client description (second turn)
    second_call_content = str(generator.calls[1][-1].content)
    assert "Sarah Martinez" in second_call_content


async def test_persona_simulator_maintains_client_description_across_turns():
    """Test that same client description is used for all turns."""
    generator = MockPersonaGenerator(
        responses=[
            {
                "client_description": "John Doe, 40, tech-savvy",
                "goal_reached": False,
                "message": "First message",
            },
            {
                "goal_reached": False,
                "message": "Second message",
            },
            {
                "goal_reached": True,
                "message": None,
            },
        ]
    )

    simulator = PersonaSimulator(generator=generator, persona="power_user", max_steps=3)

    trace = LLMTrace()
    gen = simulator(trace)

    # Turn 1
    inputs1 = await anext(gen)
    assert inputs1 == "First message"
    assert simulator._client_description == "John Doe, 40, tech-savvy"

    # Turn 2
    trace = await trace.with_interaction(
        Interaction(inputs=inputs1, outputs="Response 1")
    )
    inputs2 = await gen.asend(trace)
    assert inputs2 == "Second message"
    assert simulator._client_description == "John Doe, 40, tech-savvy"  # Same

    # Turn 3
    trace = await trace.with_interaction(
        Interaction(inputs=inputs2, outputs="Response 2")
    )
    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    # Client description should remain the same
    assert simulator._client_description == "John Doe, 40, tech-savvy"


async def test_persona_simulator_respects_max_steps():
    """Test that simulator respects max_steps limit."""
    generator = MockPersonaGenerator(
        responses=[
            {
                "client_description": "Test User",
                "goal_reached": False,
                "message": "Message 1",
            },
            {
                "goal_reached": False,
                "message": "Message 2",
            },
        ]
    )

    simulator = PersonaSimulator(
        generator=generator, persona="helpful_user", max_steps=1
    )

    trace = LLMTrace()
    gen = simulator(trace)

    # First turn
    inputs = await anext(gen)
    assert inputs == "Message 1"
    assert len(generator.calls) == 1

    # Second turn should not happen (max_steps=1)
    trace = await trace.with_interaction(Interaction(inputs=inputs, outputs="Response"))
    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    # Should still only have 1 call
    assert len(generator.calls) == 1
