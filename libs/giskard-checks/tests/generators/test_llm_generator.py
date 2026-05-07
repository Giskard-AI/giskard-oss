# libs/giskard-checks/tests/generators/test_llm_generator.py
import pytest
from giskard.checks import Interaction
from giskard.checks.generators.base import LLMGenerator

from .conftest import LLMTrace, MockGenerator


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


@pytest.mark.asyncio
async def test_llm_generator_stops_immediately_when_max_steps_zero():
    mock_gen = MockGenerator(responses=[])
    gen = LLMGenerator(generator=mock_gen, prompt="Say something.", max_steps=0)
    trace = LLMTrace()
    agen = gen(trace)
    with pytest.raises(StopAsyncIteration):
        await anext(agen)
    assert len(mock_gen.calls) == 0


@pytest.mark.asyncio
async def test_llm_generator_stops_when_message_is_none_and_goal_not_reached():
    mock_gen = MockGenerator(
        responses=[
            {"goal_reached": False, "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say something.", max_steps=3)
    trace = LLMTrace()
    agen = gen(trace)
    with pytest.raises(StopAsyncIteration):
        await anext(agen)
    assert len(mock_gen.calls) == 1
