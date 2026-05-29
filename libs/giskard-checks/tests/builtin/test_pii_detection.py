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
        mode="llm",  # Force LLM mode to preserve existing test behavior
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
        mode="llm",  # Force LLM mode to ensure we test LLM handling
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None
    assert result.message == "Check passed"  # Message defaults to "Check passed"


async def test_check_is_serialisable() -> None:
    """Test that the check can be serialised and deserialised via Pydantic."""
    from giskard.agents.generators import Generator
    from giskard.checks.core.check import Check

    check = PIIDetection(
        output="Some text.",
        categories=["email", "phone"],
        generator=Generator(model="openai/gpt-4o"),
    )
    data = check.model_dump()
    assert data["kind"] == "pii_detection"
    assert data["categories"] == ["email", "phone"]

    # Verify round-trip deserialization via the discriminated union
    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, PIIDetection)
    assert reconstructed.categories == ["email", "phone"]


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


# ============= HYBRID MODE TESTS (Pattern + LLM) =============


async def test_pattern_mode_detects_email_without_llm() -> None:
    """Test that pattern mode detects email PII without calling LLM."""
    generator = MockGenerator(passed=True, reason="No PII found")
    check = PIIDetection(
        generator=generator,
        output="Contact me at john@example.com",
        mode="pattern",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_via"] == "pattern"
    assert result.details["severity"] in ("high", "critical")
    assert result.details["confidence"] == 1.0
    assert "email" in result.details["categories_detected"]
    # Verify LLM was not called (no calls to generator)
    assert len(generator.calls) == 0


async def test_pattern_mode_detects_phone_without_llm() -> None:
    """Test that pattern mode detects phone numbers without calling LLM."""
    generator = MockGenerator(passed=True, reason="No PII found")
    check = PIIDetection(
        generator=generator,
        output="Call me at +1-555-123-4567",
        mode="pattern",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_via"] == "pattern"
    assert "phone" in result.details["categories_detected"]
    assert len(generator.calls) == 0


async def test_pattern_mode_passes_clean_content() -> None:
    """Test that pattern mode passes content without structured PII."""
    generator = MockGenerator(passed=True, reason="Clean content")
    check = PIIDetection(
        generator=generator,
        output="This is a clean response with no structured PII.",
        mode="pattern",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["detected_via"] == "pattern"
    assert result.details["severity"] == "low"
    assert result.details["confidence"] == 1.0
    assert result.details["categories_detected"] == []
    assert len(generator.calls) == 0


async def test_hybrid_mode_early_exit_on_high_severity_pii() -> None:
    """Test that hybrid mode fails immediately when high-severity PII is found."""
    generator = MockGenerator(passed=True, reason="LLM should not be called")
    check = PIIDetection(
        generator=generator,
        output="My SSN is 123-45-6789",
        mode="hybrid",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_via"] == "pattern"
    assert result.details["severity"] in ("high", "critical")
    assert "ssn" in result.details["categories_detected"]
    # Verify LLM was not called (early exit)
    assert len(generator.calls) == 0


async def test_hybrid_mode_calls_llm_for_low_severity_pii() -> None:
    """Test that hybrid mode calls LLM when patterns don't have matches.
    
    In hybrid mode, if patterns are checked but find no matches in any category,
    we proceed to LLM evaluation. The detected_via should be "llm" since no
    patterns matched (detected_via is "llm" when hybrid mode doesn't find patterns).
    """
    generator = MockGenerator(passed=True, reason="No contextual PII found")
    check = PIIDetection(
        generator=generator,
        output="General discussion about locations",
        mode="hybrid",
        categories=["name", "address"],  # Only contextual categories (no regex patterns)
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    # Pattern layer won't match these categories, so we call LLM
    # detected_via is "llm" since hybrid mode didn't find any patterns to skip LLM
    assert result.details["detected_via"] == "llm"
    # LLM should be called since patterns don't cover name/address
    assert len(generator.calls) == 1


async def test_hybrid_mode_early_exit_on_pattern_catch() -> None:
    """Test that hybrid mode exits early when high-severity PII is caught by patterns.
    
    In hybrid mode, when patterns detect high-severity PII (like emails, SSNs, etc.),
    the check fails immediately without calling the LLM (early exit optimization).
    The detected_via is "pattern" since we exited via pattern detection.
    """
    generator = MockGenerator(
        passed=False,
        reason="Response contains a name: John Smith and an email address",
    )
    check = PIIDetection(
        generator=generator,
        output="John Smith can be reached at john@example.com",
        mode="hybrid",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    # Pattern layer catches the email with high severity, so we exit early
    assert result.details["detected_via"] == "pattern"
    assert "email" in result.details["categories_detected"]
    # LLM should NOT be called (early exit optimization)
    assert len(generator.calls) == 0


async def test_severity_critical_for_ssn() -> None:
    """Test that SSN is marked as critical severity."""
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Test SSN: 123-45-6789",
        mode="pattern",
        categories=["ssn"],
    )
    result = await check.run(Trace())

    assert result.details["severity"] == "critical"


async def test_severity_critical_for_credit_card() -> None:
    """Test that credit card is marked as critical severity."""
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Card number: 4111-1111-1111-1111",
        mode="pattern",
        categories=["credit_card"],
    )
    result = await check.run(Trace())

    assert result.details["severity"] == "critical"


async def test_severity_high_for_email() -> None:
    """Test that email is marked as high severity."""
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Contact: test@example.com",
        mode="pattern",
        categories=["email"],
    )
    result = await check.run(Trace())

    assert result.details["severity"] == "high"


async def test_confidence_1_0_for_pattern_matches() -> None:
    """Test that pattern matches have confidence of 1.0 (deterministic)."""
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Email: john@example.com",
        mode="pattern",
    )
    result = await check.run(Trace())

    assert result.details["confidence"] == 1.0


async def test_pattern_caching_compiled_once() -> None:
    """Test that regex patterns are compiled once and cached."""
    import giskard.checks.judges.pii_detection as pii_module

    # Reset the cache
    pii_module._compiled_patterns_cache = None

    generator1 = MockGenerator(passed=True, reason="No PII")
    check1 = PIIDetection(
        generator=generator1,
        output="test1@example.com",
        mode="pattern",
    )
    result1 = await check1.run(Trace())

    # Verify patterns were compiled
    assert pii_module._compiled_patterns_cache is not None
    cached_patterns_1 = pii_module._compiled_patterns_cache

    generator2 = MockGenerator(passed=True, reason="No PII")
    check2 = PIIDetection(
        generator=generator2,
        output="test2@example.com",
        mode="pattern",
    )
    result2 = await check2.run(Trace())

    # Verify patterns are the same instance (cache hit)
    assert pii_module._compiled_patterns_cache is cached_patterns_1


async def test_result_details_structure_with_pattern() -> None:
    """Test that result details have expected structure for pattern mode."""
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Email: john@example.com",
        mode="pattern",
    )
    result = await check.run(Trace())

    # Verify all expected keys are present
    assert "reason" in result.details
    assert "severity" in result.details
    assert "confidence" in result.details
    assert "detected_via" in result.details
    assert "categories_detected" in result.details
    assert "inputs" in result.details

    # Verify types
    assert isinstance(result.details["severity"], str)
    assert result.details["severity"] in ("low", "medium", "high", "critical")
    assert isinstance(result.details["confidence"], float)
    assert 0.0 <= result.details["confidence"] <= 1.0
    assert isinstance(result.details["detected_via"], str)
    assert result.details["detected_via"] in ("pattern", "llm", "hybrid")
    assert isinstance(result.details["categories_detected"], list)


async def test_backward_compatibility_default_mode_hybrid() -> None:
    """Test that default mode is hybrid for backward compatibility."""
    generator = MockGenerator(passed=True, reason="Clean")
    check = PIIDetection(
        generator=generator,
        output="Clean response",
        # mode not specified - should default to "hybrid"
    )

    # Verify the mode is hybrid
    assert check.mode == "hybrid"


async def test_backward_compatibility_existing_tests_pass() -> None:
    """Test that existing functionality is preserved (LLM-only for contextual PII)."""
    generator = MockGenerator(
        passed=False,
        reason="The response contains a full name: John Smith",
    )
    check = PIIDetection(
        generator=generator,
        output="My name is John Smith",
        mode="llm",  # Force LLM mode for contextual detection
        categories=["name"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["detected_via"] == "llm"
    assert len(generator.calls) == 1


async def test_pattern_ignores_test_data_caveat() -> None:
    """Test behavior with test/example data (note: patterns are naive, may match)."""
    # This test demonstrates that patterns will match test data like test@example.com
    # Users should filter these if needed in the LLM layer (hybrid mode recommended)
    generator = MockGenerator(passed=True, reason="No PII")
    check = PIIDetection(
        generator=generator,
        output="Example code: test@example.com and 192.0.2.1",
        mode="pattern",
    )
    result = await check.run(Trace())

    # Patterns will match these (they look like real PII), but severity is lower
    # This is expected behavior - rely on LLM layer for context in hybrid mode
    assert result.status == CheckStatus.FAIL
    assert "email" in result.details["categories_detected"]
