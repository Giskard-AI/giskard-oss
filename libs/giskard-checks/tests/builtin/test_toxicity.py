"""Tests for the Toxicity LLM-based check."""

import json
from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Interaction, Toxicity, Trace
from pydantic import Field


class MockGenerator(BaseGenerator):
    """Mock generator that returns a pre-configured LLM judgement."""

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


async def test_clean_content_passes() -> None:
    """Test that clean, non-toxic content passes the check."""
    generator = MockGenerator(passed=True, reason="No toxic content detected.")
    check = Toxicity(
        generator=generator,
        output="Here is a helpful and respectful response to your question.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "No toxic content detected."
    assert len(generator.calls) == 1


async def test_toxic_content_fails() -> None:
    """Test that toxic content fails the check."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains hate speech targeting a demographic group.",
    )
    check = Toxicity(
        generator=generator,
        output="[Toxic content here]",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert "hate speech" in (result.details["reason"] or "")
    assert len(generator.calls) == 1


async def test_output_extracted_from_trace() -> None:
    """Test that output is extracted from trace when not provided directly."""
    generator = MockGenerator(passed=True, reason="Clean response.")
    check = Toxicity(generator=generator)
    interaction = Interaction(
        inputs={"query": "Tell me about history."},
        outputs={"response": "Here is a factual account of historical events."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1
    # Verify the output was passed to the prompt
    assert "inputs" in result.details
    assert "output" in result.details["inputs"]


async def test_prompt_includes_full_trace_including_prior_turns() -> None:
    """Test that the rendered prompt includes the full trace (not only the output).

    This guards the contextual-toxicity edge case where a short answer like "Yes"
    must be evaluated against harmful content earlier in the conversation.
    """
    generator = MockGenerator(passed=True, reason="Clean response.")
    check = Toxicity(generator=generator)

    trace = Trace(
        interactions=[
            Interaction(
                inputs={"user": "Do you agree with <hateful claim>?"},
                outputs={"assistant": "..."},
            ),
            Interaction(
                inputs={"user": "Do you agree?"},
                outputs={"assistant": "Yes"},
            ),
        ]
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1
    assert len(generator.calls[0]) >= 1
    prompt = generator.calls[0][0].content
    assert isinstance(prompt, str)
    assert "<TRACE>" in prompt
    assert "</TRACE>" in prompt
    assert "Do you agree with <hateful claim>?" in prompt


async def test_custom_output_key() -> None:
    """Test extraction using a custom output_key JSONPath."""
    generator = MockGenerator(passed=True, reason="Safe content.")
    check = Toxicity(
        generator=generator,
        output_key="trace.last.outputs.message",
    )
    interaction = Interaction(
        inputs={"query": "Hello"},
        outputs={"message": "Hello! How can I help you today?"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "Hello! How can I help you today?"


async def test_category_filtering_passed_to_template() -> None:
    """Test that specified categories are included in the template inputs."""
    generator = MockGenerator(passed=True, reason="No hate speech detected.")
    check = Toxicity(
        generator=generator,
        output="A normal response.",
        categories=["hate_speech", "harassment"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["categories"] == ["hate_speech", "harassment"]


async def test_default_categories_used_when_not_specified() -> None:
    """Test that all default categories are used when categories is not provided."""
    generator = MockGenerator(passed=True, reason="No toxicity found.")
    check = Toxicity(
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
        "hate_speech",
        "harassment",
        "threats",
        "self_harm",
        "sexual_content",
        "violence",
    ]:
        assert expected in categories


async def test_violence_category_fails() -> None:
    """Test that violent content is detected and fails."""
    generator = MockGenerator(
        passed=False,
        reason="Response contains instructions for violence.",
    )
    check = Toxicity(
        generator=generator,
        output="[Violent content here]",
        categories=["violence"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["categories"] == ["violence"]


async def test_none_reason_is_handled() -> None:
    """Test that a None reason from the LLM is handled gracefully."""
    generator = MockGenerator(passed=True, reason=None)
    check = Toxicity(
        generator=generator,
        output="Clean response.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None


async def test_direct_output_overrides_trace() -> None:
    """Test that a directly provided output takes precedence over the trace."""
    generator = MockGenerator(passed=True, reason="Clean.")
    check = Toxicity(
        generator=generator,
        output="Directly provided text.",
    )
    interaction = Interaction(
        inputs={"query": "test"},
        outputs={"response": "Trace output that should be ignored."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "Directly provided text."


async def test_check_is_serialisable() -> None:
    """Test that the check can be serialised and deserialised via Pydantic."""
    from giskard.agents.generators import Generator
    from giskard.checks.core.check import Check

    check = Toxicity(
        output="Some text.",
        categories=["hate_speech"],
        generator=Generator(model="openai/gpt-4o"),
    )
    data = check.model_dump()
    assert data["kind"] == "toxicity"
    assert data["categories"] == ["hate_speech"]

    # Verify round-trip deserialization via the discriminated union
    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, Toxicity)
    assert reconstructed.categories == ["hate_speech"]
