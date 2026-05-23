"""Tests for the PIIDetection LLM-based check."""

import json
from collections.abc import Sequence
from typing import Any, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Interaction, PIIDetection, Trace
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field


class MockGenerator(BaseGenerator):
    """Mock generator that returns a pre-configured LLM judgement."""

    passed: bool
    reason: str | None
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=json.dumps(
                            {"passed": self.passed, "reason": self.reason}
                        )
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


async def test_clean_content_passes() -> None:
    """Test that clean, non-PII content passes the check."""
    generator = MockGenerator(passed=True, reason="No PII detected in the response.")
    check = PIIDetection(
        generator=generator,
        output="Here is a helpful and respectful response to your question.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "No PII detected in the response."
    assert len(generator.calls) == 1


async def test_pii_content_fails() -> None:
    """Test that content with PII fails the check."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains an email address: john@example.com",
    )
    check = PIIDetection(
        generator=generator,
        output="You can reach me at john@example.com",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert "email" in (result.details["reason"] or "").lower()
    assert len(generator.calls) == 1


async def test_output_extracted_from_trace() -> None:
    """Test that output is extracted from trace when not provided directly."""
    generator = MockGenerator(passed=True, reason="No PII detected.")
    check = PIIDetection(generator=generator)
    interaction = Interaction(
        inputs={"query": "What is your email?"},
        outputs={"response": "I cannot provide personal email addresses."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1
    # Verify the output was passed to the prompt
    assert "inputs" in result.details
    assert "output" in result.details["inputs"]


async def test_prompt_includes_full_trace_including_prior_turns() -> None:
    """Test that the rendered prompt includes the full trace (not only the output).

    This guards the edge case where a response must be evaluated in context
    of earlier PII-related discussion.
    """
    generator = MockGenerator(passed=True, reason="No PII in final response.")
    check = PIIDetection(generator=generator)

    trace = Trace(
        interactions=[
            Interaction(
                inputs={"user": "What is your phone number?"},
                outputs={"assistant": "..."},
            ),
            Interaction(
                inputs={"user": "Can you confirm?"},
                outputs={"assistant": "I cannot share personal information."},
            ),
        ]
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1
    assert len(generator.calls[0]) >= 1
    prompt = generator.calls[0][0].transcript
    assert isinstance(prompt, str)
    assert "<TRACE>" in prompt
    assert "</TRACE>" in prompt
    assert "What is your phone number?" in prompt


async def test_custom_output_key() -> None:
    """Test extraction using a custom output_key JSONPath."""
    generator = MockGenerator(passed=True, reason="No PII found.")
    check = PIIDetection(
        generator=generator,
        output_key="trace.last.outputs.message",
    )
    interaction = Interaction(
        inputs={"query": "Hello"},
        outputs={"message": "Hello! I am here to help, no personal data shared."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "Hello! I am here to help, no personal data shared."


async def test_category_filtering_passed_to_template() -> None:
    """Test that specified categories are included in the template inputs."""
    generator = MockGenerator(passed=True, reason="No email or phone detected.")
    check = PIIDetection(
        generator=generator,
        output="A normal response.",
        categories=["email", "phone"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["categories"] == ["email", "phone"]


async def test_default_categories_used_when_not_specified() -> None:
    """Test that all default categories are used when categories is not provided."""
    generator = MockGenerator(passed=True, reason="No PII found.")
    check = PIIDetection(
        generator=generator,
        output="A safe response.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    categories = result.details["inputs"]["categories"]
    assert isinstance(categories, list)
    assert len(categories) > 0
    # Verify the default categories are all present
    for expected in [
        "email",
        "phone",
        "ssn",
        "credit_card",
        "ip_address",
        "name",
        "address",
        "medical",
        "financial",
    ]:
        assert expected in categories


async def test_email_category_fails() -> None:
    """Test that email PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains an email address.",
    )
    check = PIIDetection(
        generator=generator,
        output="Contact me at user@domain.com",
        categories=["email"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["email"]


async def test_pattern_mode_affects_prompt_inputs() -> None:
    """Test that mode='pattern' affects prompt inputs/rendering."""
    generator = MockGenerator(passed=True, reason="No patterns detected.")
    check = PIIDetection(
        generator=generator,
        output="A normal response.",
        mode="pattern",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["mode"] == "pattern"


async def test_llm_mode_affects_prompt_inputs() -> None:
    """Test that mode='llm' affects prompt inputs/rendering."""
    generator = MockGenerator(passed=True, reason="No contextual PII detected.")
    check = PIIDetection(
        generator=generator,
        output="A normal response.",
        mode="llm",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["mode"] == "llm"


async def test_hybrid_mode_affects_prompt_inputs() -> None:
    """Test that mode='hybrid' affects prompt inputs/rendering."""
    generator = MockGenerator(passed=True, reason="No PII detected with hybrid detection.")
    check = PIIDetection(
        generator=generator,
        output="A normal response.",
        mode="hybrid",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["mode"] == "hybrid"


async def test_direct_output_overrides_trace() -> None:
    """Test that a directly provided output takes precedence over the trace."""
    generator = MockGenerator(passed=True, reason="Clean.")
    check = PIIDetection(
        generator=generator,
        output="Directly provided text with no PII.",
    )
    interaction = Interaction(
        inputs={"query": "test"},
        outputs={"response": "Trace output that should be ignored with user@example.com."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "Directly provided text with no PII."


async def test_none_reason_is_handled() -> None:
    """Test that a None reason from the LLM is handled gracefully."""
    generator = MockGenerator(passed=True, reason=None)
    check = PIIDetection(
        generator=generator,
        output="Clean response.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None


async def test_check_is_serialisable() -> None:
    """Test that the check can be serialised and deserialised via Pydantic."""
    from giskard.agents.generators import Generator
    from giskard.checks.core.check import Check

    check = PIIDetection(
        output="Some text.",
        categories=["email", "phone"],
        mode="hybrid",
        generator=Generator(model="openai/gpt-4o"),
    )
    data = check.model_dump()
    assert data["kind"] == "pii_detection"
    assert data["categories"] == ["email", "phone"]
    assert data["mode"] == "hybrid"

    # Verify round-trip deserialization via the discriminated union
    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, PIIDetection)
    assert reconstructed.categories == ["email", "phone"]
    assert reconstructed.mode == "hybrid"


async def test_phone_number_detection() -> None:
    """Test that phone number PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains a phone number.",
    )
    check = PIIDetection(
        generator=generator,
        output="Call me at 555-123-4567",
        categories=["phone"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["phone"]


async def test_ssn_detection() -> None:
    """Test that SSN PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains a social security number.",
    )
    check = PIIDetection(
        generator=generator,
        output="My SSN is 123-45-6789",
        categories=["ssn"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["ssn"]


async def test_credit_card_detection() -> None:
    """Test that credit card PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains a credit card number.",
    )
    check = PIIDetection(
        generator=generator,
        output="Card: 4111-1111-1111-1111",
        categories=["credit_card"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["credit_card"]


async def test_ip_address_detection() -> None:
    """Test that IP address PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains an IP address.",
    )
    check = PIIDetection(
        generator=generator,
        output="Server at 192.168.1.1",
        categories=["ip_address"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["ip_address"]


async def test_name_detection() -> None:
    """Test that name PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains a full name.",
    )
    check = PIIDetection(
        generator=generator,
        output="My name is John Smith",
        categories=["name"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["name"]


async def test_address_detection() -> None:
    """Test that address PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains a physical address.",
    )
    check = PIIDetection(
        generator=generator,
        output="I live at 123 Main Street, Springfield, IL 62701",
        categories=["address"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["address"]


async def test_medical_info_detection() -> None:
    """Test that medical info PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains medical information.",
    )
    check = PIIDetection(
        generator=generator,
        output="I have been diagnosed with diabetes",
        categories=["medical"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["medical"]


async def test_financial_info_detection() -> None:
    """Test that financial info PII is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains financial information.",
    )
    check = PIIDetection(
        generator=generator,
        output="My bank account is 987654321",
        categories=["financial"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["financial"]
