import pytest

from giskard_checks.checks import Conformity
from giskard_checks.core import Check, InteractionResult
from tests.test_utils.mock_generator import MockGenerator


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Rule conformed"),
        (False, "Rule not conformed"),
    ],
)
async def test_conformity_pass_fail_roundtrip(passed: bool, reason: str):
    """Test basic conformity check with direct rule string."""
    interaction = InteractionResult(
        inputs="This is a test question.", outputs="This is a test response."
    )

    rule = "The response should be polite and professional."

    check = Conformity(
        rule=rule,
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["rule"] == rule
    assert res1.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res1.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res1.details["inputs"]["interaction_metadata"] == repr(None)
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Conformity)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["rule"] == rule
    assert res2.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res2.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res2.details["inputs"]["interaction_metadata"] == repr(None)
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Rule conformed with templating"),
        (False, "Rule not conformed with templating"),
    ],
)
async def test_conformity_with_rule_templating(passed: bool, reason: str):
    """Test conformity with Jinja2 rule templating."""
    interaction = InteractionResult(
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris.",
        metadata={"category": "geography", "difficulty": "easy"},
    )

    # Rule with Jinja2 templating to access interaction fields
    rule = "The response should answer the question '{{ inputs }}' and be appropriate for {{ metadata.category }} questions."

    check = Conformity(
        rule=rule,
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    # The rule should be formatted with the actual interaction values
    expected_formatted_rule = "The response should answer the question 'What is the capital of France?' and be appropriate for geography questions."
    assert res1.details["inputs"]["rule"] == expected_formatted_rule
    assert res1.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res1.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res1.details["inputs"]["interaction_metadata"] == repr(interaction.metadata)
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Conformity)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["rule"] == expected_formatted_rule
    assert res2.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res2.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res2.details["inputs"]["interaction_metadata"] == repr(interaction.metadata)
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Rule conformed with complex data"),
        (False, "Rule not conformed with complex data"),
    ],
)
async def test_conformity_with_complex_data(passed: bool, reason: str):
    """Test conformity with complex data structures and templating."""
    interaction = InteractionResult(
        inputs={"question": "What is the capital of France?", "context": "geography"},
        outputs={"answer": "Paris", "confidence": 0.95, "source": "wikipedia"},
        metadata={"model": "gpt-4", "timestamp": "2024-01-01", "category": "geography"},
    )

    # Rule accessing nested fields in the interaction
    rule = "The response should have high confidence (>=0.9) and include the answer for {{ inputs.question }}."

    check = Conformity(
        rule=rule,
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    # The rule should be formatted with the actual interaction values
    expected_formatted_rule = "The response should have high confidence (>=0.9) and include the answer for What is the capital of France?."
    assert res1.details["inputs"]["rule"] == expected_formatted_rule
    assert res1.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res1.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res1.details["inputs"]["interaction_metadata"] == repr(interaction.metadata)
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Conformity)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["rule"] == expected_formatted_rule
    assert res2.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res2.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res2.details["inputs"]["interaction_metadata"] == repr(interaction.metadata)
    assert res2.message == reason


async def test_conformity_without_metadata():
    """Test conformity without metadata."""
    interaction = InteractionResult(inputs="Simple question", outputs="Simple answer")

    rule = "The response should be concise."

    check = Conformity(
        rule=rule,
        generator=MockGenerator(
            outputs='{"passed": true, "reason": "Response is concise"}'
        ),
    )

    res = await check.run(interaction)
    assert res.passed is True
    assert res.details["reason"] == "Response is concise"
    assert res.details["inputs"]["rule"] == rule
    assert res.details["inputs"]["interaction_inputs"] == repr(interaction.inputs)
    assert res.details["inputs"]["interaction_outputs"] == repr(interaction.outputs)
    assert res.details["inputs"]["interaction_metadata"] == repr(interaction.metadata)


async def test_conformity_serialization():
    """Test that Conformity can be serialized and deserialized properly."""
    rule = "Test rule for serialization"
    check = Conformity(rule=rule)

    # Serialize
    serialized = check.model_dump()
    assert serialized["rule"] == rule
    assert serialized["kind"] == "conformity"

    # Deserialize
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Conformity)
    assert deserialized.rule == rule
