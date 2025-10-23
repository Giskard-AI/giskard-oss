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
        inputs="This is a test question.", outputs="This is a test response."
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ inputs }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert (
        res1.details["inputs"]["template"]
        == "Analyze this text: This is a test question.. Is it in French"
    )
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert (
        res2.details["inputs"]["template"]
        == "Analyze this text: This is a test question.. Is it in French"
    )
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_inline_prompt_check_outputs_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck accessing outputs."""
    interaction = Interaction(
        inputs="This is a test question.", outputs="This is a test response."
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ outputs }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert (
        res1.details["inputs"]["template"]
        == "Analyze this text: This is a test response.. Is it in French"
    )
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert (
        res2.details["inputs"]["template"]
        == "Analyze this text: This is a test response.. Is it in French"
    )
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_inline_prompt_check_metadata_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck accessing metadata."""
    interaction = Interaction(
        inputs="This is a test question.",
        outputs="This is a test response.",
        metadata={"category": "test", "language": "en"},
    )

    check = InlinePromptCheck(
        template_content="Analyze this text: {{ inputs }} with category {{ metadata.category }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert (
        res1.details["inputs"]["template"]
        == "Analyze this text: This is a test question. with category test. Is it in French"
    )
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, InlinePromptCheck)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert (
        res2.details["inputs"]["template"]
        == "Analyze this text: This is a test question. with category test. Is it in French"
    )
    assert res2.message == reason
