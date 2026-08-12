from giskard.checks import CheckStatus, Correctness, Interaction, Trace

from ..testing_utils import LLMTrace
from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Matches reference")
    correctness = Correctness(
        generator=generator,
        agent_description="Q&A bot",
        answer="Paris.",
        reference_answer="The capital of France is Paris.",
    )
    result = await correctness.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Matches reference"

    assert len(generator.calls) == 1
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Contradicts reference")
    correctness = Correctness(
        generator=generator,
        agent_description="Q&A bot",
        answer="Lyon.",
        reference_answer="The capital of France is Paris.",
    )
    result = await correctness.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Contradicts reference"

    assert len(generator.calls) == 1


async def test_inputs_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    correctness = Correctness(generator=generator)
    interaction = Interaction(
        inputs="Capital of France?",
        outputs="Paris.",
        metadata={
            "reference_answer": "Paris is the capital of France.",
        },
    )
    trace = LLMTrace(
        annotations={"description": "Factual assistant"},
        interactions=[interaction],
    )
    result = await correctness.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None

    assert len(generator.calls) == 1
    prompt = generator.calls[0][0].transcript
    assert "<AGENT DESCRIPTION>\nFactual assistant\n</AGENT DESCRIPTION>" in prompt
    assert f"<CONVERSATION>\n{trace._repr_prompt_()}\n</CONVERSATION>" in prompt
    assert "<AGENT ANSWER>\nParis.\n</AGENT ANSWER>" in prompt
    assert (
        "<REFERENCE ANSWER>\nParis is the capital of France.\n</REFERENCE ANSWER>"
        in prompt
    )
