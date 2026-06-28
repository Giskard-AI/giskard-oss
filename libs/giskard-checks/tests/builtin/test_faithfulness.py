"""Unit tests for the Faithfulness check."""

from giskard.checks import CheckStatus, Faithfulness, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_faithful_answer_passes() -> None:
    generator = MockGenerator(passed=True, reason="Answer accurately represents source")
    check = Faithfulness(
        generator=generator,
        answer="Python was created in 1991.",
        source="Python was first released in 1991 by Guido van Rossum.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["reason"] == "Answer accurately represents source"


async def test_unfaithful_answer_fails() -> None:
    generator = MockGenerator(passed=False, reason="Answer distorts the source")
    check = Faithfulness(
        generator=generator,
        answer="Python was created to replace Java.",
        source="Python was created as a general-purpose scripting language.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.failed


async def test_llm_called_once() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Faithfulness(
        generator=generator,
        answer="Paris is the capital of France.",
        source="France's capital city is Paris.",
    )
    await check.run(Trace())

    assert len(generator.calls) == 1


async def test_answer_and_source_extracted_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Faithfulness(generator=generator)
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Summarize",
            outputs="The sky is blue.",
            metadata={"context": "The sky appears blue due to Rayleigh scattering."},
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert "The sky is blue" in result.details["inputs"]["answer"]
    assert "Rayleigh scattering" in result.details["inputs"]["source"]


async def test_list_source_joined() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Faithfulness(
        generator=generator,
        answer="Python is used widely.",
        source=["Python is a popular language.", "It is used in many domains."],
    )
    result = await check.run(Trace())

    assert result.passed
    assert "Python is a popular language." in result.details["inputs"]["source"]
    assert "It is used in many domains." in result.details["inputs"]["source"]


async def test_direct_values_take_priority_over_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Faithfulness(
        generator=generator,
        answer="Direct answer",
        source="Direct source",
    )
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Trace input",
            outputs="Trace answer",
            metadata={"context": "Trace source"},
        )
    )
    result = await check.run(trace)

    assert result.passed
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert result.details["inputs"]["source"] == "Direct source"
