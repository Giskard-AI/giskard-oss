from typing import cast

from giskard.checks import CheckStatus, Contradiction, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success() -> None:
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
    assert len(generator.calls) == 1
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
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
    assert len(generator.calls) == 1


async def test_direct_answer_and_context_are_passed_to_judge() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        answer="Direct answer",
        context=["Context 1", "Context 2"],
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert result.details["inputs"]["context"] == "Context 1\nContext 2"
    assert len(generator.calls) == 1


async def test_single_string_context_is_passed_to_judge() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        answer="The Eiffel Tower is in Paris.",
        context="The Eiffel Tower is in Paris.",
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "The Eiffel Tower is in Paris."
    assert result.details["inputs"]["context"] == "The Eiffel Tower is in Paris."


async def test_prompt_tolerates_refusals_without_conflicting_factual_claims() -> None:
    generator = MockGenerator(passed=True, reason="Refusal is not a contradiction")
    contradiction = Contradiction(
        generator=generator,
        answer="I can't help with harmful requests.",
        context=["ZephyrBank accounts are available to eligible applicants."],
    )

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert "incorrectly frames the request as harmful" in prompt
    assert "does not also make a factual claim that conflicts" in prompt


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
        metadata={"context": ["Paris is the capital of France."]},
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Mock reason."
    assert result.details["inputs"]["answer"] == str(
        {"response": "The Eiffel Tower is in Paris."}
    )
    assert "Paris is the capital of France." in result.details["inputs"]["context"]


async def test_custom_answer_and_context_keys() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        target_key="trace.interactions[0].outputs.response",
        context_key="trace.interactions[0].metadata.documents",
    )
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
        metadata={"documents": ["The Eiffel Tower is in Paris."]},
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "The Eiffel Tower is in Paris."
    assert result.details["inputs"]["context"] == "The Eiffel Tower is in Paris."


async def test_direct_values_take_priority_over_trace() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        answer="Direct answer",
        context=["Direct context"],
    )
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "Trace answer"},
        metadata={"context": ["Trace context"]},
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert result.details["inputs"]["context"] == "Direct context"


async def test_list_context_is_joined_without_python_repr_artifacts() -> None:
    """A list[str] context must reach the judge prompt as readable text.

    Regression test for the bug where get_inputs() rendered a list context via
    bare str(), producing the Python repr (e.g. "['doc 1', 'doc 2']") instead of
    the documents themselves. Documents containing apostrophes are especially
    revealing: str(repr) mixes quote styles and escapes the apostrophe, which
    should never appear in the actual prompt text.
    """
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        answer="The Eiffel Tower is in Paris, and it's a landmark.",
        context=[
            "Paris is the capital of France.",
            "It's located in Europe.",
        ],
    )

    result = await contradiction.run(Trace())

    context_str = cast(str, result.details["inputs"]["context"])
    assert context_str == "Paris is the capital of France.\nIt's located in Europe."
    assert "[" not in context_str
    assert "]" not in context_str
    assert '\\"' not in context_str
    assert "\\'" not in context_str


async def test_list_answer_from_trace_is_joined_without_python_repr_artifacts() -> None:
    """A list-valued answer extracted via answer_key must not leak Python repr."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(
        generator=generator,
        target_key="trace.last.metadata.answer_parts",
        context="Paris is the capital of France.",
    )
    interaction = Interaction(
        inputs={"query": "Where is Paris?"},
        outputs={"response": "unused"},
        metadata={
            "answer_parts": [
                "Paris is the capital of France.",
                "It's located in Europe.",
            ]
        },
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    answer_str = cast(str, result.details["inputs"]["answer"])
    assert answer_str == "Paris is the capital of France.\nIt's located in Europe."
    assert "[" not in answer_str
    assert "]" not in answer_str
    assert '\\"' not in answer_str
    assert "\\'" not in answer_str


async def test_missing_trace_values_return_error_without_judge() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(generator=generator)

    result = await contradiction.run(Trace())

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "answer key" in (result.message or "")
    assert "trace.last.outputs" in (result.message or "")
    assert len(generator.calls) == 0


async def test_missing_context_returns_error_without_judge() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
    )

    result = await contradiction.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "context key" in (result.message or "")
    assert "trace.last.metadata.context" in (result.message or "")
    assert len(generator.calls) == 0


async def test_not_does_not_invert_missing_context_error() -> None:
    from giskard.checks import Not

    generator = MockGenerator(passed=True, reason="Mock reason.")
    contradiction = Contradiction(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
    )

    result = await Not(check=contradiction).run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "context key" in (result.message or "")
    assert len(generator.calls) == 0
