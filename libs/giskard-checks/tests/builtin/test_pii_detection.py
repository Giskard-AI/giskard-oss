"""Tests for the PIIDetection check."""

import json
from typing import Any, override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Check, CheckStatus, Interaction, PIIDetection, Trace
from pydantic import Field, ValidationError


class MockGenerator(BaseGenerator):
    """Mock generator that returns a pre-configured PII judgement."""

    passed: bool
    reason: str | None
    calls: list[list[Message]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        self.calls.append(messages)
        return Response(
            message=Message(
                role="assistant",
                content=json.dumps({"passed": self.passed, "reason": self.reason}),
            ),
            finish_reason="stop",
        )


async def test_pattern_mode_detects_email() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        output="Reach me at jane.doe@example.com.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_categories"] == ["email"]
    assert result.details["pattern_matches"]["email"] == ["jane.doe@example.com"]
    assert generator.calls == []


async def test_pattern_mode_detects_credit_card_with_major_issuer_pattern() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        categories=["credit_card"],
        output="Use card 4111 1111 1111 1111 for the payment test.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_categories"] == ["credit_card"]
    assert result.details["pattern_matches"]["credit_card"] == [
        "4111 1111 1111 1111"
    ]


async def test_pattern_mode_passes_on_clean_output() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        output="The customer has been a member since 2020.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["detected_categories"] == []
    assert generator.calls == []


async def test_pattern_mode_does_not_flag_plain_nine_digit_number_as_ssn() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        categories=["ssn"],
        output="The internal identifier is 123456789.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["pattern_matches"] == {}


async def test_pattern_mode_does_not_flag_invalid_ip_address() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        categories=["ip_address"],
        output="The reported host was 999.999.999.999 during testing.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["pattern_matches"] == {}


async def test_hybrid_short_circuits_on_structured_pii() -> None:
    generator = MockGenerator(passed=True, reason="This should not be used.")
    check = PIIDetection(
        generator=generator,
        mode="hybrid",
        output="Call me at 555-123-4567.",
        categories=["phone", "name"],
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_categories"] == ["phone"]
    assert generator.calls == []


async def test_hybrid_uses_llm_for_contextual_pii() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The response reveals a person's full name.",
    )
    check = PIIDetection(
        generator=generator,
        mode="hybrid",
        output="The customer is John Smith from support.",
        categories=["name"],
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "The response reveals a person's full name."
    assert len(generator.calls) == 1
    prompt = generator.calls[0][0].content
    assert isinstance(prompt, str)
    assert "name" in prompt


async def test_custom_output_key() -> None:
    generator = MockGenerator(passed=True, reason="No contextual PII detected.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        output_key="trace.last.outputs.message",
    )
    interaction = Interaction(
        inputs={"query": "Share status"},
        outputs={"message": "No personal information disclosed."},
    )

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "No personal information disclosed."


async def test_llm_mode_passes_selected_categories_to_prompt() -> None:
    generator = MockGenerator(
        passed=True,
        reason="No selected PII categories were disclosed.",
    )
    check = PIIDetection(
        generator=generator,
        mode="llm",
        output="The user has been a member since 2020.",
        categories=["name", "address"],
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["categories"] == ["name", "address"]
    assert len(generator.calls) == 1


def test_pattern_mode_rejects_contextual_categories() -> None:
    with pytest.raises(
        ValidationError,
        match="Pattern mode only supports structured PII categories",
    ):
        _ = PIIDetection(mode="pattern", categories=["name"])


async def test_default_pattern_categories_used() -> None:
    generator = MockGenerator(passed=True, reason="Unused in pattern mode.")
    check = PIIDetection(
        generator=generator,
        mode="pattern",
        output="The answer is clean.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["categories"] == [
        "email",
        "phone",
        "ssn",
        "credit_card",
        "ip_address",
    ]


async def test_check_is_serialisable() -> None:
    from giskard.agents.generators import Generator

    check = PIIDetection(
        output="The user is Jane Doe.",
        mode="llm",
        categories=["name"],
        generator=Generator(model="openai/gpt-4o"),
    )

    data = check.model_dump()
    assert data["kind"] == "pii_detection"
    assert data["mode"] == "llm"
    assert data["categories"] == ["name"]

    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, PIIDetection)
    assert reconstructed.mode == "llm"
    assert reconstructed.categories == ["name"]
