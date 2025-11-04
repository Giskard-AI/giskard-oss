"""Tests for InlinePromptCheck."""

import pytest
from giskard.checks.checks.judge import LLMJudge
from giskard.checks.core.interaction_result import InteractionResult

from tests.test_utils.mock_generator import MockGenerator


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_judge_check_direct_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck."""
    interaction = InteractionResult(
        inputs="This is a test question.", outputs="This is a test response."
    )

    check = LLMJudge(
        prompt="Analyze this text: {{ interaction.inputs }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = LLMJudge.model_validate(serialized)
    assert isinstance(deserialized, LLMJudge)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Text is in French"),
        (False, "Text is not in French"),
    ],
)
async def test_judge_check_outputs_instantiation(passed: bool, reason: str):
    """Test direct instantiation of InlinePromptCheck accessing outputs."""
    interaction = InteractionResult(
        inputs="This is a test question.", outputs="This is a test response."
    )

    check = LLMJudge(
        prompt="Analyze this text: {{ interaction.outputs }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = LLMJudge.model_validate(serialized)
    assert isinstance(deserialized, LLMJudge)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
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
    interaction = InteractionResult(
        inputs="This is a test question.",
        outputs="This is a test response.",
        metadata={"category": "test", "language": "en"},
    )

    check = LLMJudge(
        prompt="Analyze this text: {{ interaction.inputs }} with category {{ interaction.metadata.category }}. Is it in French",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = LLMJudge.model_validate(serialized)
    assert isinstance(deserialized, LLMJudge)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.message == reason
