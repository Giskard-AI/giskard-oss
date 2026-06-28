"""Unit tests for the Hallucination check."""

from giskard.checks import CheckStatus, Hallucination, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_accurate_answer_passes() -> None:
    generator = MockGenerator(passed=True, reason="No hallucinations detected")
    check = Hallucination(
        generator=generator,
        answer="Python was created in 1991.",
        context="Python was first released in 1991.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["reason"] == "No hallucinations detected"


async def test_hallucinated_answer_fails() -> None:
    generator = MockGenerator(passed=False, reason="Fabricated statistic detected")
    check = Hallucination(
        generator=generator,
        answer="Python has over 500 million users worldwide.",
        context="Python is a widely used programming language.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.failed


async def test_llm_called_once() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(
        generator=generator,
        answer="Water boils at 100°C at sea level.",
    )
    await check.run(Trace())

    assert len(generator.calls) == 1


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(generator=generator)
    trace = await Trace.from_interactions(
        Interaction(
            inputs="What year was Python created?",
            outputs="Python was created in 1991.",
            metadata={"context": "Python was first released in 1991."},
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert "1991" in result.details["inputs"]["answer"]
    assert "1991" in result.details["inputs"]["context"]


async def test_direct_context_takes_priority() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(
        generator=generator,
        answer="Direct answer",
        context="Direct context",
    )
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Trace input",
            outputs="Trace answer",
            metadata={"context": "Trace context"},
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert result.details["inputs"]["context"] == "Direct context"


async def test_list_context_joined() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(
        generator=generator,
        answer="Python is popular.",
        context=["Python is widely used.", "It was created in 1991."],
    )
    result = await check.run(Trace())

    assert result.passed
    assert "Python is widely used." in result.details["inputs"]["context"]
    assert "It was created in 1991." in result.details["inputs"]["context"]


async def test_works_without_context() -> None:
    """Check should not crash when no context is provided or resolvable."""
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(
        generator=generator,
        answer="The Eiffel Tower is in Paris.",
    )
    result = await check.run(Trace())

    assert result.passed
