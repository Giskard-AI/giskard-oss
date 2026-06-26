from giskard.checks import CheckStatus, Contradiction, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success_when_no_contradiction() -> None:
    generator = MockGenerator(
        passed=True,
        reason="The extra detail is not contradicted by the context",
    )
    contradiction = Contradiction(
        generator=generator,
        answer="The Eiffel Tower is in Paris and is popular with tourists.",
        context=["The Eiffel Tower is in Paris."],
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.PASS
    assert (
        result.details["reason"]
        == "The extra detail is not contradicted by the context"
    )


async def test_run_returns_failure_on_clear_contradiction() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The answer places the Eiffel Tower in Tokyo, contradicting the context.",
    )
    contradiction = Contradiction(
        generator=generator,
        answer="The Eiffel Tower is in Tokyo.",
        context=["The Eiffel Tower is in Paris."],
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert (
        result.details["reason"]
        == "The answer places the Eiffel Tower in Tokyo, contradicting the context."
    )


async def test_prompt_tolerates_omissions_and_additions() -> None:
    generator = MockGenerator(passed=True, reason="No clear contradiction")
    contradiction = Contradiction(
        generator=generator,
        answer="I don't know the full answer, but it may be in Paris.",
        context=["The Eiffel Tower is in Paris and was completed in 1889."],
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert "Omissions and additions are allowed" in prompt
    assert "unsupported additions without direct conflict" in prompt
    assert "passes unless it clearly contradicts" in prompt


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    contradiction = Contradiction(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
        metadata={"context": ["Paris is the capital of France."]},
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == str(
        {"response": "The Eiffel Tower is in Paris."}
    )
    assert "Paris is the capital of France." in result.details["inputs"]["context"]
