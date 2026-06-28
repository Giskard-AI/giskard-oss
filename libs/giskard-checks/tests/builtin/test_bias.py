"""Unit tests for the Bias check."""

from giskard.checks import Bias, CheckStatus, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_unbiased_output_passes() -> None:
    generator = MockGenerator(passed=True, reason="No bias detected")
    check = Bias(
        generator=generator,
        output="Software engineers come from diverse backgrounds and skill sets.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["reason"] == "No bias detected"


async def test_biased_output_fails() -> None:
    generator = MockGenerator(passed=False, reason="Gender stereotyping detected")
    check = Bias(
        generator=generator,
        output="Women are naturally less suited for engineering roles.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.failed


async def test_llm_called_once() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Bias(
        generator=generator,
        output="All people deserve equal opportunities.",
    )
    await check.run(Trace())

    assert len(generator.calls) == 1


async def test_output_extracted_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Bias(generator=generator)
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Describe a doctor",
            outputs="Doctors are dedicated professionals.",
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert "Doctors are dedicated professionals" in result.details["inputs"]["output"]


async def test_attribute_filtering_forwarded() -> None:
    generator = MockGenerator(passed=False, reason="Racial bias detected")
    check = Bias(
        generator=generator,
        output="Some races are inherently smarter.",
        protected_attributes=["race"],
    )
    result = await check.run(Trace())

    assert result.failed
    assert result.details["inputs"]["protected_attributes"] == ["race"]


async def test_context_forwarded_when_key_provided() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Bias(
        generator=generator,
        context_key="trace.last.inputs",
    )
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Describe a nurse",
            outputs="Nurses are compassionate caregivers.",
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert result.details["inputs"]["context"] != ""


async def test_no_context_key_gives_empty_context() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Bias(
        generator=generator,
        output="Everyone is treated equally here.",
    )
    result = await check.run(Trace())

    assert result.passed
    assert result.details["inputs"]["context"] == ""
