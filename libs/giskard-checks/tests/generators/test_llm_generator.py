# libs/giskard-checks/tests/generators/test_llm_generator.py
import pytest
from giskard.checks import InputGenerationException, Interaction, Scenario
from giskard.checks.generators.base import LLMGenerator, LLMGeneratorOutput
from pydantic import BaseModel

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


def test_llm_generator_output_is_generic():
    class MyModel(BaseModel):
        content: str

    output = LLMGeneratorOutput[MyModel](
        goal_reached=False, message=MyModel(content="hi")
    )
    assert output.message is not None
    assert output.message.content == "hi"


def test_llm_generator_output_has_schema_issue():
    output = LLMGeneratorOutput[str](goal_reached=False, schema_issue="no string field")
    assert output.schema_issue == "no string field"
    assert output.message is None


def test_llm_generator_output_schema_issue_defaults_to_none():
    output = LLMGeneratorOutput[str](goal_reached=False, message="hello")
    assert output.schema_issue is None


class UserMessage(BaseModel):
    role: str = "user"
    content: str


@pytest.mark.asyncio
async def test_llm_generator_produces_base_model_when_input_type_provided():
    mock_gen = MockGenerator(
        responses=[
            {
                "goal_reached": False,
                "schema_issue": None,
                "message": {"role": "user", "content": "Hello"},
            },
            {"goal_reached": True, "schema_issue": None, "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say hello.", max_steps=5)
    trace = LLMTrace()
    agen = gen(trace, input_type=UserMessage)
    msg = await anext(agen)
    assert isinstance(msg, UserMessage)
    assert msg.content == "Hello"


@pytest.mark.asyncio
async def test_llm_generator_raises_on_schema_issue():
    mock_gen = MockGenerator(
        responses=[
            {"goal_reached": False, "schema_issue": "no string field", "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say something.", max_steps=3)
    trace = LLMTrace()
    agen = gen(trace, input_type=UserMessage)
    with pytest.raises(InputGenerationException, match="schema issue: no string field"):
        await anext(agen)


@pytest.mark.asyncio
async def test_llm_generator_parses_structured_response_using_input_type_schema():
    # with_output(LLMGeneratorOutput[UserMessage]) controls response parsing —
    # the LLM response JSON is deserialized into a UserMessage instance.
    # Schema injection into the prompt requires {{ _instr_output }} in the template.
    mock_gen = MockGenerator(
        responses=[
            {
                "goal_reached": False,
                "schema_issue": None,
                "message": {"role": "user", "content": "Hi"},
            },
            {"goal_reached": True, "schema_issue": None, "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say something.", max_steps=3)
    trace = LLMTrace()
    agen = gen(trace, input_type=UserMessage)
    msg = await anext(agen)
    assert isinstance(msg, UserMessage)
    assert msg.content == "Hi"
    assert len(mock_gen.calls) == 1


@pytest.mark.asyncio
async def test_llm_generator_str_output_unchanged_without_input_type():
    mock_gen = MockGenerator(
        responses=[
            {"goal_reached": False, "schema_issue": None, "message": "Hello there"},
            {"goal_reached": True, "schema_issue": None, "message": None},
        ]
    )
    gen = LLMGenerator(generator=mock_gen, prompt="Say hello.", max_steps=5)
    trace = LLMTrace()
    agen = gen(trace)
    msg = await anext(agen)
    assert msg == "Hello there"
    assert isinstance(msg, str)


# --- End-to-end: Scenario + LLMGenerator with as_template ---


@pytest.mark.asyncio
async def test_full_chain_llm_generator_produces_user_message_for_base_model_target():
    mock_gen = MockGenerator(
        responses=[
            {
                "goal_reached": False,
                "schema_issue": None,
                "message": {"role": "user", "content": "Tell me about your product"},
            },
            {"goal_reached": True, "schema_issue": None, "message": None},
        ]
    )
    received_inputs: list[UserMessage] = []

    def agent_target(inputs: UserMessage) -> str:
        received_inputs.append(inputs)
        return f"Response to: {inputs.content}"

    llm_gen = LLMGenerator(
        generator=mock_gen,
        prompt="Ask about the product.\n{{ _instr_output }}",
        max_steps=5,
        as_template=True,
    )
    scenario = Scenario(name="test").interact(inputs=llm_gen, outputs=agent_target)

    result = await scenario.run()
    assert result.passed
    assert len(received_inputs) == 1
    assert isinstance(received_inputs[0], UserMessage)
    assert received_inputs[0].content == "Tell me about your product"
    assert len(mock_gen.calls) == 2
    for call in mock_gen.calls:
        assert (
            str(LLMGeneratorOutput[UserMessage].model_json_schema())
            in call[0].transcript
        )


@pytest.mark.asyncio
async def test_full_chain_llm_generator_raises_on_schema_issue():
    class _NoStringLikeField(BaseModel):
        content: int

    mock_gen = MockGenerator(
        responses=[
            {
                "goal_reached": False,
                "schema_issue": "no string-like field in schema",
                "message": None,
            },
        ]
    )

    def agent_target(inputs: _NoStringLikeField) -> str:
        return f"Response to: {inputs.content}"

    llm_gen = LLMGenerator(
        generator=mock_gen,
        prompt="Ask about the product.\n{{ _instr_output }}",
        max_steps=5,
        as_template=True,
    )
    scenario = Scenario(name="test").interact(inputs=llm_gen, outputs=agent_target)

    with pytest.raises(
        InputGenerationException, match="schema issue: no string-like field in schema"
    ):
        await scenario.run()

    assert len(mock_gen.calls) == 1
    assert (
        str(LLMGeneratorOutput[_NoStringLikeField].model_json_schema())
        in mock_gen.calls[0][0].transcript
    )


@pytest.mark.asyncio
async def test_full_chain_llm_generator_does_not_render_template_when_as_template_false():
    """When as_template=False, Jinja2 syntax in the prompt must not be evaluated.

    This is a security guard: user-controlled content in the prompt should not
    be rendered as a template, preventing prompt injection via template execution.
    """
    mock_gen = MockGenerator(
        responses=[
            {
                "goal_reached": False,
                "schema_issue": None,
                "message": {"role": "user", "content": "Tell me about your product"},
            },
            {"goal_reached": True, "schema_issue": None, "message": None},
        ]
    )

    def agent_target(inputs: UserMessage) -> str:
        return f"Response to: {inputs.content}"

    llm_gen = LLMGenerator(
        generator=mock_gen,
        prompt="Ask about the product.\n{{ _instr_output }}",
        max_steps=5,
        as_template=False,
    )
    scenario = Scenario(name="test").interact(inputs=llm_gen, outputs=agent_target)

    await scenario.run()

    # With as_template=False the raw literal "{{ _instr_output }}" must appear
    # in the prompt message — it must NOT have been replaced by the schema JSON.
    assert len(mock_gen.calls) >= 1
    first_message_content = mock_gen.calls[0][0].transcript
    assert "{{ _instr_output }}" in first_message_content
    assert (
        str(LLMGeneratorOutput[UserMessage].model_json_schema())
        not in first_message_content
    )
