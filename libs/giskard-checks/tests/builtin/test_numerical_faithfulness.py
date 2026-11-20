from typing import cast

from giskard.checks import CheckStatus, Interaction, NumericalFaithfulness, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Numbers are faithful")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Revenue grew 12.5% to 10M users.",
        context=["Revenue grew 12.5% to 10,000,000 users."],
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Numbers are faithful"

    assert len(generator.calls) == 1
    # The prompt comes from the template file, so we check that the call was made
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(
        passed=False, reason="Answer says 1.25% but context says 12.5%"
    )
    check = NumericalFaithfulness(
        generator=generator,
        answer="Revenue grew 1.25% to 1M users.",
        context=["Revenue grew 12.5% to 10,000,000 users."],
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Answer says 1.25% but context says 12.5%"

    assert len(generator.calls) == 1


async def test_prompt_focuses_on_numerical_faithfulness() -> None:
    generator = MockGenerator(passed=True, reason="No numbers to check")
    check = NumericalFaithfulness(
        generator=generator,
        answer="The capital of France is Paris.",
        context=["Paris is the capital of France."],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert "numerical hallucination" in prompt
    assert "Percentage and decimal equivalence" in prompt
    assert "Magnitude suffixes" in prompt
    assert "Textual and digit forms" in prompt
    assert "Omissions" in prompt
    # The numerical-specific equivalences from the prompt should reach the judge
    assert "five million" in prompt


STRICT_DERIVED_RULE = "Treat every such derived number as a mismatch"
LENIENT_DERIVED_RULE = "recompute each derived value from the context numbers yourself"


async def test_strict_by_default_forbids_derived_values() -> None:
    generator = MockGenerator(passed=False, reason="15M is not in the context")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Total revenue was 15M.",
        context=["Product revenue was 10M and services revenue was 5M."],
    )
    assert check.strict is True
    await check.run(Trace())

    prompt = generator.calls[0][0].transcript
    assert "## Derived Values" in prompt
    assert STRICT_DERIVED_RULE in prompt


async def test_non_strict_drops_derived_values_ban() -> None:
    generator = MockGenerator(passed=True, reason="15M = 10M + 5M")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Total revenue was 15M.",
        context=["Product revenue was 10M and services revenue was 5M."],
        strict=False,
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert "## Derived Values" in prompt
    assert STRICT_DERIVED_RULE not in prompt
    assert LENIENT_DERIVED_RULE in prompt


async def test_strict_prompt_omits_lenient_rule() -> None:
    """The lenient-mode phrase must be absent from the strict prompt."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Total revenue was 15M.",
        context=["Product revenue was 10M and services revenue was 5M."],
    )
    await check.run(Trace())

    prompt = generator.calls[0][0].transcript
    assert LENIENT_DERIVED_RULE not in prompt
    assert STRICT_DERIVED_RULE in prompt


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(generator=generator)
    interaction = Interaction(
        inputs={"query": "How much revenue?"},
        outputs={"response": "Revenue grew 12.5% to 10M users."},
        metadata={"context": ["Revenue grew 12.5% to 10,000,000 users."]},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Mock reason."

    assert len(generator.calls) == 1
    # Verify that answer and context were extracted from trace
    assert "inputs" in result.details
    assert "answer" in result.details["inputs"]
    assert "context" in result.details["inputs"]
    # answer_key defaults to "trace.last.outputs" which returns the entire dict
    assert result.details["inputs"]["answer"] == str(
        {"response": "Revenue grew 12.5% to 10M users."}
    )
    assert "10,000,000 users." in result.details["inputs"]["context"]


async def test_direct_answer_and_context() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Revenue: 12.5%",
        context=["Context 1", "Context 2"],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["answer"] == "Revenue: 12.5%"
    assert "Context 1" in result.details["inputs"]["context"]
    assert "Context 2" in result.details["inputs"]["context"]


async def test_direct_answer_and_single_string_context() -> None:
    """Test that context can be a single string instead of a list."""
    generator = MockGenerator(
        passed=True, reason="Answer is faithful to single context string"
    )
    check = NumericalFaithfulness(
        generator=generator,
        answer="Revenue grew 12.5% to 10M users.",
        context="Revenue grew 12.5% to 10,000,000 users in the last quarter.",
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["answer"] == "Revenue grew 12.5% to 10M users."
    assert (
        result.details["inputs"]["context"]
        == "Revenue grew 12.5% to 10,000,000 users in the last quarter."
    )
    assert len(generator.calls) == 1


async def test_custom_keys() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        target_key="trace.interactions[0].outputs.response",
        context_key="trace.interactions[0].metadata.documents",
    )
    interaction = Interaction(
        inputs={"query": "What is the revenue?"},
        outputs={"response": "Revenue is 10M."},
        metadata={"documents": ["Revenue is 10,000,000.", "Another document"]},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Revenue is 10M."
    # Context should contain the documents
    context_str = cast(str, result.details["inputs"]["context"])
    assert "Revenue is 10,000,000." in context_str or "Another document" in context_str


async def test_answer_priority_over_trace() -> None:
    """Test that direct answer takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Direct answer takes priority",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Trace answer"},
        metadata={"context": ["Supporting context"]},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer takes priority"


async def test_context_priority_over_trace() -> None:
    """Test that direct context takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        context=["Direct context"],
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"context": ["Trace context"]},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["context"] == "Direct context"


async def test_empty_string_context_is_preserved() -> None:
    """Test that an empty string context is preserved and does not fall back to trace."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Some answer",
        context="",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"context": ["Trace context"]},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["context"] == ""


async def test_empty_context() -> None:
    """Test behavior with empty context."""
    generator = MockGenerator(passed=False, reason="No context provided")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Some answer",
        context=[],
    )
    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["context"] == ""


async def test_list_context_is_joined_without_python_repr_artifacts() -> None:
    """A list[str] context must reach the judge prompt as readable text."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        answer="Revenue grew 12.5% to 10M users.",
        context=[
            "Revenue grew 12.5% in Q1.",
            "It's 10,000,000 users total.",
        ],
    )
    result = await check.run(Trace())

    context_str = cast(str, result.details["inputs"]["context"])
    assert context_str == "Revenue grew 12.5% in Q1.\nIt's 10,000,000 users total."
    assert "[" not in context_str
    assert "]" not in context_str
    assert '\\"' not in context_str
    assert "\\'" not in context_str


async def test_list_answer_from_trace_is_joined_without_python_repr_artifacts() -> None:
    """A list-valued answer extracted via answer_key must not leak Python repr."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(
        generator=generator,
        target_key="trace.last.metadata.answer_parts",
        context="Revenue grew 12.5% to 10,000,000 users.",
    )
    interaction = Interaction(
        inputs={"query": "What is the revenue?"},
        outputs={"response": "unused"},
        metadata={
            "answer_parts": [
                "Revenue grew 12.5%.",
                "It's 10,000,000 users total.",
            ]
        },
    )
    result = await check.run(Trace(interactions=[interaction]))

    answer_str = cast(str, result.details["inputs"]["answer"])
    assert answer_str == "Revenue grew 12.5%.\nIt's 10,000,000 users total."
    assert "[" not in answer_str
    assert "]" not in answer_str
    assert '\\"' not in answer_str
    assert "\\'" not in answer_str


async def test_missing_answer_in_trace() -> None:
    """Missing answer key returns ERROR without invoking the judge."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(generator=generator)
    # Empty trace - no interactions
    result = await check.run(Trace())

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "answer key" in (result.message or "")
    assert "trace.last.outputs" in (result.message or "")
    assert len(generator.calls) == 0


async def test_missing_context_in_trace() -> None:
    """Missing context key returns ERROR without invoking the judge."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        # No context in metadata
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "context key" in (result.message or "")
    assert "trace.last.metadata.context" in (result.message or "")
    assert len(generator.calls) == 0


async def test_not_does_not_invert_missing_context_error() -> None:
    from giskard.checks import Not

    generator = MockGenerator(passed=True, reason="Mock reason.")
    check = NumericalFaithfulness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
    )

    result = await Not(check=check).run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "context key" in (result.message or "")
    assert len(generator.calls) == 0


async def test_using_trace_last_property() -> None:
    """Test that demonstrates using trace.last instead of trace.interactions[-1]."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    interaction1 = Interaction(
        inputs={"query": "First question"},
        outputs={"response": "First answer"},
        metadata={"context": ["Context 1"]},
    )
    interaction2 = Interaction(
        inputs={"query": "Second question"},
        outputs={"response": "Second answer"},
        metadata={"context": ["Context 2"]},
    )
    trace = Trace(interactions=[interaction1, interaction2])

    # Verify trace.last works and equals the last interaction
    assert trace.last is not None
    assert trace.last == interaction2
    assert trace.last.outputs == {"response": "Second answer"}
    assert trace.last == trace.interactions[-1]

    # Use trace.last to verify the check uses the last interaction
    check = NumericalFaithfulness(
        generator=generator,
        target_key="trace.last.outputs.response",
        context_key="trace.last.metadata.context",
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    # Verify it extracted from the last interaction
    assert result.details["inputs"]["answer"] == "Second answer"
    assert "Context 2" in result.details["inputs"]["context"]


async def test_trace_last_with_empty_trace() -> None:
    """Empty trace has no last interaction; the check returns ERROR."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    trace = Trace()

    # Verify trace.last returns None for empty trace
    assert trace.last is None

    check = NumericalFaithfulness(generator=generator)
    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "answer key" in (result.message or "")
    assert len(generator.calls) == 0
