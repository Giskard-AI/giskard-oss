"""Tests for InlinePromptCheck."""

import pytest

from giskard_checks.checks.llm_check import Check, InlinePromptCheck
from giskard_checks.core.interactions import Interaction
from tests.test_utils.mock_generator import MockGenerator


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_inline_prompt_check_direct_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck."""
    interaction = Interaction(
        input="This is a test question.", output="This is a test response."
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ text }}. Is it in French",
        template_input={"text": "This is a test question."},
        generator=MockGenerator(
            output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["text"] == "This is a test question."
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["text"] == "This is a test question."
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_inline_prompt_check_keys_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck."""
    interaction = Interaction(
        input="This is a test question.", output="This is a test response."
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ text }}. Is it in French",
        template_input_keys={"text": "$.output"},
        generator=MockGenerator(
            output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["text"] == "This is a test response."
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["text"] == "This is a test response."
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_inline_prompt_check_mixed_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck."""
    interaction = Interaction(
        input="This is a test question.", output="This is a test response."
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ text }}. Is it in French",
        template_input={"text": "This is a test question."},
        template_input_keys={"text": "$.output"},
        generator=MockGenerator(
            output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["text"] == "This is a test question."
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        output=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["text"] == "This is a test question."
    assert res2.message == reason
