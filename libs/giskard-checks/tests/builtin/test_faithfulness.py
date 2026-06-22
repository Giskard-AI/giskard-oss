"""Tests for the Faithfulness LLM-based check."""

from typing import cast

import pytest
from giskard.checks import (
    Check,
    CheckStatus,
    Faithfulness,
    FaithfulnessCheckResult,
    Interaction,
    Trace,
)
from pydantic import ValidationError

from ..testing_utils import MockGenerator


async def test_faithful_answer_passes() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 0.95,
                "passed": True,
                "reason": "The answer accurately represents the source.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="The policy allows refunds within 30 days.",
        source="Customers may request a refund within 30 days of purchase.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["reason"] == "The answer accurately represents the source."
    assert result.details["score"] == 0.95
    assert result.details["threshold"] == 0.8
    assert result.metrics[0].name == "faithfulness"
    assert result.metrics[0].value == 0.95
    assert len(generator.calls) == 1


async def test_distorted_answer_fails() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 0.2,
                "passed": False,
                "reason": "The answer changes the refund window from 30 days to 90 days.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="The policy allows refunds within 90 days.",
        source="Customers may request a refund within 30 days of purchase.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.details["reason"] == (
        "The answer changes the refund window from 30 days to 90 days."
    )
    assert result.details["score"] == 0.2
    assert result.metrics[0].value == 0.2


async def test_partial_faithfulness_fails_below_threshold() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 0.6,
                "passed": True,
                "reason": "Some claims are supported, but one claim is not in the source.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="The policy allows refunds within 30 days and covers shipping.",
        source="Customers may request a refund within 30 days of purchase.",
        threshold=0.8,
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.details["score"] == 0.6
    assert result.details["passed"] is True


async def test_configurable_threshold_allows_partial_score() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 0.6,
                "passed": True,
                "reason": "Mostly faithful with one minor unsupported detail.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="The policy allows refunds within 30 days and covers shipping.",
        source="Customers may request a refund within 30 days of purchase.",
        threshold=0.5,
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["score"] == 0.6
    assert result.details["threshold"] == 0.5


async def test_answer_and_source_from_trace() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 1.0,
                "passed": True,
                "reason": "The answer matches the source.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer_key="trace.last.outputs.answer",
        source_key="trace.last.metadata.source",
    )
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Summarize the policy",
            outputs={"answer": "Refunds are available within 30 days."},
            metadata={"source": "Refunds are available within 30 days of purchase."},
        )
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Refunds are available within 30 days."
    assert (
        result.details["inputs"]["source"]
        == "Refunds are available within 30 days of purchase."
    )


async def test_source_list_is_joined_for_prompt() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 0.9,
                "passed": True,
                "reason": "The answer is supported by the source material.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="Refunds are available within 30 days.",
        source=[
            "Refunds are available within 30 days of purchase.",
            "Exchanges are available within 60 days.",
        ],
    )

    result = await check.run(Trace())

    source = cast(str, result.details["inputs"]["source"])
    assert "Refunds are available within 30 days of purchase." in source
    assert "Exchanges are available within 60 days." in source
    assert "\n\n" in source


async def test_prompt_includes_answer_source_and_threshold() -> None:
    generator = MockGenerator(
        responses=[
            {
                "score": 1.0,
                "passed": True,
                "reason": "Faithful.",
            }
        ]
    )
    check = Faithfulness(
        generator=generator,
        answer="The release is scheduled for July.",
        source="The release is scheduled for July.",
        threshold=0.9,
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert isinstance(prompt, str)
    assert "<ANSWER>" in prompt
    assert "<SOURCE MATERIAL>" in prompt
    assert "The release is scheduled for July." in prompt
    assert "Minimum passing score: 0.9" in prompt


def test_threshold_validation() -> None:
    with pytest.raises(ValidationError):
        Faithfulness(threshold=1.5)


def test_faithfulness_output_validation() -> None:
    with pytest.raises(ValidationError):
        FaithfulnessCheckResult(score=1.5, passed=True, reason=None)


def test_faithfulness_is_exported() -> None:
    assert Faithfulness.__name__ == "Faithfulness"


def test_faithfulness_serialization_roundtrip() -> None:
    check = Faithfulness(
        answer_key="trace.last.outputs.answer",
        source_key="trace.last.metadata.source",
        threshold=0.9,
    )

    data = check.model_dump()
    restored = Check.model_validate(data)

    assert data["kind"] == "faithfulness"
    assert isinstance(restored, Faithfulness)
    assert restored.answer_key == "trace.last.outputs.answer"
    assert restored.source_key == "trace.last.metadata.source"
    assert restored.threshold == 0.9
