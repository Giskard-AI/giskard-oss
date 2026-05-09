"""Tests for the Bias LLM-based check.

Acceptance criteria from issue #2366:
- Unbiased content passes
- Biased content fails
- Attribute filtering works
- Reason is provided for bias detection
"""

import json
from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Bias, CheckStatus, Interaction, Trace
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


# ── Acceptance criteria 1: unbiased content passes ────────────────────────────


async def test_unbiased_content_passes() -> None:
    """Unbiased response should pass the check."""
    generator = MockGenerator(passed=True, reason="No bias detected.")
    check = Bias(
        generator=generator,
        output="Software engineers come from all backgrounds and demographics.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "No bias detected."


# ── Acceptance criteria 2: biased content fails ───────────────────────────────


async def test_gender_biased_content_fails() -> None:
    """Gender-biased output should fail with a reason."""
    generator = MockGenerator(
        passed=False,
        reason="Response stereotypes nursing as a female profession.",
    )
    check = Bias(
        generator=generator,
        output="Women tend to be more nurturing and make better nurses.",
        protected_attributes=["gender"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] is not None
    assert len(result.details["reason"]) > 0


async def test_racial_bias_fails() -> None:
    """Racially biased output should fail."""
    generator = MockGenerator(
        passed=False,
        reason="Response attributes negative characteristics to a racial group.",
    )
    check = Bias(
        generator=generator,
        output="[Racially biased content]",
        protected_attributes=["race"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL


async def test_age_bias_fails() -> None:
    """Age-biased output should fail."""
    generator = MockGenerator(
        passed=False,
        reason="Assumes older employees cannot learn new technology — ageist stereotype.",
    )
    check = Bias(
        generator=generator,
        output="Older employees usually struggle to adapt to modern software tools.",
        protected_attributes=["age"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL


# ── Acceptance criteria 3: attribute filtering works ─────────────────────────


async def test_single_attribute_filter_passed_to_template() -> None:
    """Specified protected_attributes should be forwarded to the prompt template."""
    generator = MockGenerator(passed=True, reason="No gender bias detected.")
    check = Bias(
        generator=generator,
        output="All candidates are assessed on merit.",
        protected_attributes=["gender"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["protected_attributes"] == ["gender"]


async def test_multiple_attribute_filters() -> None:
    """Multiple protected_attributes should all be forwarded."""
    generator = MockGenerator(passed=True, reason="No bias found.")
    check = Bias(
        generator=generator,
        output="A fair and balanced response.",
        protected_attributes=["gender", "race", "age"],
    )
    result = await check.run(Trace())

    assert set(result.details["inputs"]["protected_attributes"]) == {
        "gender",
        "race",
        "age",
    }


async def test_default_attributes_used_when_none_specified() -> None:
    """When protected_attributes is None, all defaults should be used."""
    generator = MockGenerator(passed=True, reason="No bias found.")
    check = Bias(generator=generator, output="A balanced response.")
    result = await check.run(Trace())

    attrs = result.details["inputs"]["protected_attributes"]
    assert isinstance(attrs, list)
    for expected in ["gender", "race", "age", "religion"]:
        assert expected in attrs


# ── Acceptance criteria 4: reason is provided ─────────────────────────────────


async def test_reason_provided_on_failure() -> None:
    """A non-None reason must be provided when bias is detected."""
    generator = MockGenerator(
        passed=False,
        reason="Response uses exclusionary language based on religion.",
    )
    check = Bias(
        generator=generator,
        output="[Biased content]",
        protected_attributes=["religion"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] is not None
    assert len(result.details["reason"]) > 0


async def test_none_reason_handled_gracefully() -> None:
    """A None reason from the LLM should not raise an error."""
    generator = MockGenerator(passed=True, reason=None)
    check = Bias(generator=generator, output="A clean response.")
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None


# ── context_key support ───────────────────────────────────────────────────────


async def test_context_key_extracted_from_trace() -> None:
    """context_key should extract context and pass it to the template."""
    generator = MockGenerator(passed=False, reason="Response endorses biased premise.")
    check = Bias(
        generator=generator,
        context_key="trace.last.inputs",
        protected_attributes=["age"],
    )
    interaction = Interaction(
        inputs={"query": "Aren't older workers less productive?"},
        outputs={"response": "Yes, that is generally true."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["context"] is not None


async def test_context_is_none_when_context_key_not_set() -> None:
    """When context_key is None, context template variable should be None."""
    generator = MockGenerator(passed=True, reason="No bias.")
    check = Bias(generator=generator, output="A fair response.")
    result = await check.run(Trace())

    assert result.details["inputs"]["context"] is None


# ── trace extraction ──────────────────────────────────────────────────────────


async def test_output_extracted_from_trace_via_key() -> None:
    """Output should be extracted from trace using the default key."""
    generator = MockGenerator(passed=True, reason="Unbiased.")
    check = Bias(generator=generator)
    interaction = Interaction(
        inputs={"query": "Describe an engineer."},
        outputs={"response": "Engineers can be anyone regardless of background."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert "output" in result.details["inputs"]


async def test_full_trace_included_in_prompt_for_contextual_bias() -> None:
    """The rendered prompt must include full trace history for contextual evaluation."""
    generator = MockGenerator(passed=False, reason="Contextual gender bias.")
    check = Bias(generator=generator)

    trace = Trace(
        interactions=[
            Interaction(
                inputs={"user": "Don't you think men are better leaders?"},
                outputs={"assistant": "..."},
            ),
            Interaction(
                inputs={"user": "So we should promote men first?"},
                outputs={"assistant": "That could be a reasonable approach."},
            ),
        ]
    )
    await check.run(trace)

    prompt = generator.calls[0][0].content
    assert isinstance(prompt, str)
    assert "<TRACE>" in prompt
    assert "men are better leaders" in prompt


# ── serialisation ─────────────────────────────────────────────────────────────


async def test_check_is_serialisable() -> None:
    """Check should round-trip through Pydantic model_dump / model_validate."""
    from giskard.agents.generators import Generator
    from giskard.checks.core.check import Check

    check = Bias(
        protected_attributes=["gender", "race"],
        context_key="trace.last.inputs",
        generator=Generator(model="openai/gpt-4o"),
    )
    data = check.model_dump()
    assert data["kind"] == "bias"
    assert data["protected_attributes"] == ["gender", "race"]
    assert data["context_key"] == "trace.last.inputs"

    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, Bias)
    assert reconstructed.protected_attributes == ["gender", "race"]
