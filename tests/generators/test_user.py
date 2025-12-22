import json
from typing import override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import Interaction, Trace, UserSimulator
from pydantic import Field


class MockGenerator(BaseGenerator):
    responses: list[str | None]
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
                content=json.dumps(
                    {
                        "message": self.responses[self.index],
                        "goal_reached": self.responses[self.index] is None,
                    }
                ),
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


async def test_user_simulator_returns_messages_until_goal_reached():
    generator = MockGenerator(responses=["Hello, how are you?", None])
    user_simulator = UserSimulator(
        generator=generator, instructions="Greet the chatbot", max_steps=2
    )

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
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
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
        generator.calls[1][-1].content
    )


async def test_user_simulator_returns_messages_until_max_steps():
    generator = MockGenerator(responses=["Hello, how are you?", "I'm good too", None])
    user_simulator = UserSimulator(
        generator=generator, instructions="Greet the chatbot", max_steps=1
    )

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert len(generator.calls) == 1
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
        generator.calls[0][-1].content
    )

    trace = await trace.with_interaction(
        Interaction(inputs=inputs, outputs="I'm good and you?")
    )
    with pytest.raises(StopAsyncIteration):
        _ = await gen.asend(trace)

    assert len(generator.calls) == 1


async def test_user_simulatorm_multiple_steps():
    generator = MockGenerator(responses=["Hello, how are you?", "I'm good too", None])
    user_simulator = UserSimulator(
        generator=generator, instructions="Greet the chatbot"
    )

    trace = LLMTrace()
    gen = user_simulator(trace)
    inputs = await anext(gen)
    assert inputs == "Hello, how are you?"
    assert len(generator.calls) == 1
    assert _wrap_in_xml_tag(trace._repr_prompt_(), "history") in str(
        generator.calls[0][-1].content
    )
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
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
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
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
    assert _wrap_in_xml_tag(user_simulator.instructions, "instructions") in str(
        generator.calls[2][-1].content
    )
