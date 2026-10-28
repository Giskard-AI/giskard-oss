from typing import cast

from giskard.checks import CheckStatus, Groundedness, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Answer is grounded in context")
    groundedness = Groundedness(
        generator=generator,
        answer="The Eiffel Tower is in Paris.",
        context=["Paris is the capital of France.", "The Eiffel Tower is a landmark."],
    )
    result = await groundedness.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Answer is grounded in context"

    assert len(generator.calls) == 1
    # The prompt comes from the template file, so we check that the call was made
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Answer is not grounded in context")
    groundedness = Groundedness(
        generator=generator,
        answer="The Eiffel Tower is in Tokyo.",
        context=["Paris is the capital of France.", "The Eiffel Tower is a landmark."],
    )
    result = await groundedness.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Answer is not grounded in context"

    assert len(generator.calls) == 1


async def test_prompt_allows_explicit_refusals_without_context_support() -> None:
    generator = MockGenerator(passed=True, reason="Explicit refusal")
    groundedness = Groundedness(
        generator=generator,
        answer="I can't help with that request.",
        context=["The Eiffel Tower is a landmark in Paris."],
    )
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert "do not flag that refusal as ungrounded" in prompt
    assert "Treat the refusal itself as allowed" in prompt
    assert "Any additional factual claims" in prompt


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
        metadata={"context": ["Paris is the capital of France."]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Mock reason."

    assert len(generator.calls) == 1
    # Verify that answer and context were extracted from trace
    assert "inputs" in result.details
    assert "answer" in result.details["inputs"]
    assert "context" in result.details["inputs"]
    # answer_key defaults to "trace.last.outputs" which returns the entire dict
    assert result.details["inputs"]["answer"] == str(
        {"response": "The Eiffel Tower is in Paris."}
    )
    assert "Paris is the capital of France." in result.details["inputs"]["context"]


async def test_direct_answer_and_context() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        answer="Direct answer",
        context=["Context 1", "Context 2"],
    )
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert "Context 1" in result.details["inputs"]["context"]
    assert "Context 2" in result.details["inputs"]["context"]


async def test_direct_answer_and_single_string_context() -> None:
    """Test that context can be a single string instead of a list."""
    generator = MockGenerator(
        passed=True, reason="Answer is grounded in single context string"
    )
    groundedness = Groundedness(
        generator=generator,
        answer="The Eiffel Tower is in Paris.",
        context="Paris is the capital of France. The Eiffel Tower is a famous landmark located there.",
    )
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["answer"] == "The Eiffel Tower is in Paris."
    assert (
        result.details["inputs"]["context"]
        == "Paris is the capital of France. The Eiffel Tower is a famous landmark located there."
    )
    assert len(generator.calls) == 1


async def test_custom_keys() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        target_key="trace.interactions[0].outputs.response",
        context_key="trace.interactions[0].metadata.documents",
    )
    interaction = Interaction(
        inputs={"query": "What is AI?"},
        outputs={"response": "AI is artificial intelligence."},
        metadata={"documents": ["Document about AI", "Another document"]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "AI is artificial intelligence."
    # Context should contain the documents
    context_str = cast(str, result.details["inputs"]["context"])
    assert "Document about AI" in context_str or "Another document" in context_str


async def test_answer_priority_over_trace() -> None:
    """Test that direct answer takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        answer="Direct answer takes priority",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Trace answer"},
        metadata={"context": ["Supporting context"]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer takes priority"


async def test_context_priority_over_trace() -> None:
    """Test that direct context takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        context=["Direct context"],
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"context": ["Trace context"]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["context"] == "Direct context"


async def test_empty_string_context_is_preserved() -> None:
    """Test that an empty string context is preserved and does not fall back to trace."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        answer="Some answer",
        context="",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"context": ["Trace context"]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["context"] == ""


async def test_empty_context() -> None:
    """Test behavior with empty context."""
    generator = MockGenerator(passed=False, reason="No context provided")
    groundedness = Groundedness(
        generator=generator,
        answer="Some answer",
        context=[],
    )
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["context"] == ""


async def test_list_context_is_joined_without_python_repr_artifacts() -> None:
    """A list[str] context must reach the judge prompt as readable text.

    Regression test for the bug where get_inputs() rendered a list context via
    bare str(), producing the Python repr (e.g. "['doc 1', 'doc 2']") instead of
    the documents themselves. Documents containing apostrophes are especially
    revealing: str(repr) mixes quote styles and escapes the apostrophe, which
    should never appear in the actual prompt text.
    """
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
        generator=generator,
        answer="The Eiffel Tower is in Paris, and it's a landmark.",
        context=[
            "Paris is the capital of France.",
            "It's located in Europe.",
        ],
    )
    result = await groundedness.run(Trace())

    context_str = cast(str, result.details["inputs"]["context"])
    assert context_str == "Paris is the capital of France.\nIt's located in Europe."
    assert "[" not in context_str
    assert "]" not in context_str
    assert '\\"' not in context_str
    assert "\\'" not in context_str


async def test_list_answer_from_trace_is_joined_without_python_repr_artifacts() -> None:
    """A list-valued answer extracted via answer_key must not leak Python repr."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(
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
    result = await groundedness.run(Trace(interactions=[interaction]))

    answer_str = cast(str, result.details["inputs"]["answer"])
    assert answer_str == "Paris is the capital of France.\nIt's located in Europe."
    assert "[" not in answer_str
    assert "]" not in answer_str
    assert '\\"' not in answer_str
    assert "\\'" not in answer_str


async def test_missing_answer_in_trace() -> None:
    """Missing answer key returns ERROR without invoking the judge."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(generator=generator)
    # Empty trace - no interactions
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "answer key" in (result.message or "")
    assert "trace.last.outputs" in (result.message or "")
    assert len(generator.calls) == 0


async def test_missing_context_in_trace() -> None:
    """Missing context key returns ERROR without invoking the judge."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        # No context in metadata
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "context key" in (result.message or "")
    assert "trace.last.metadata.context" in (result.message or "")
    assert len(generator.calls) == 0


async def test_not_does_not_invert_missing_context_error() -> None:
    from giskard.checks import Not

    generator = MockGenerator(passed=True, reason="Mock reason.")
    groundedness = Groundedness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
    )

    result = await Not(check=groundedness).run(Trace(interactions=[interaction]))

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

    # Use trace.last to verify the groundedness check uses the last interaction
    groundedness = Groundedness(
        generator=generator,
        target_key="trace.last.outputs.response",
        context_key="trace.last.metadata.context",
    )
    result = await groundedness.run(trace)

    assert result.status == CheckStatus.PASS
    # Verify it extracted from the last interaction
    assert result.details["inputs"]["answer"] == "Second answer"
    assert "Context 2" in result.details["inputs"]["context"]


async def test_trace_last_with_empty_trace() -> None:
    """Empty trace has no last interaction; Groundedness returns ERROR."""
    generator = MockGenerator(passed=True, reason="Mock reason.")
    trace = Trace()

    # Verify trace.last returns None for empty trace
    assert trace.last is None

    groundedness = Groundedness(generator=generator)
    result = await groundedness.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert "answer key" in (result.message or "")
    assert len(generator.calls) == 0
