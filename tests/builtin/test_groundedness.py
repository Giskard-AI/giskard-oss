import json
from typing import cast, override

from giskard.agents.chat import Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.checks import CheckStatus, Groundedness, Interaction, Trace
from pydantic import Field


class MockGenerator(BaseGenerator):
    passed: bool
    reason: str | None
    calls: list[list[Message]] = Field(default_factory=list)

    @override
    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        self.calls.append(messages)
        return Response(
            message=Message(
                role="assistant",
                content=json.dumps({"passed": self.passed, "reason": self.reason}),
            ),
            finish_reason="stop",
        )


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


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    groundedness = Groundedness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
        metadata={"context": ["Paris is the capital of France."]},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None

    assert len(generator.calls) == 1
    # Verify that answer and context were extracted from trace
    assert "inputs" in result.details
    assert "answer" in result.details["inputs"]
    assert "context" in result.details["inputs"]
    # answer_key defaults to "trace.interactions[-1].outputs" which returns the entire dict
    assert result.details["inputs"]["answer"] == str(
        {"response": "The Eiffel Tower is in Paris."}
    )
    assert "Paris is the capital of France." in result.details["inputs"]["context"]


async def test_direct_answer_and_context() -> None:
    generator = MockGenerator(passed=True, reason=None)
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
    generator = MockGenerator(passed=True, reason=None)
    groundedness = Groundedness(
        generator=generator,
        answer_key="trace.interactions[0].outputs.response",
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
    generator = MockGenerator(passed=True, reason=None)
    groundedness = Groundedness(
        generator=generator,
        answer="Direct answer takes priority",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Trace answer"},
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer takes priority"


async def test_context_priority_over_trace() -> None:
    """Test that direct context takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason=None)
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
    assert result.details["inputs"]["context"] == "['Direct context']"


async def test_empty_string_context_is_preserved() -> None:
    """Test that an empty string context is preserved and does not fall back to trace."""
    generator = MockGenerator(passed=True, reason=None)
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
    assert result.details["inputs"]["context"] == "[]"


async def test_missing_answer_in_trace() -> None:
    """Test behavior when answer is not found in trace."""
    generator = MockGenerator(passed=True, reason=None)
    groundedness = Groundedness(generator=generator)
    # Empty trace - no interactions
    result = await groundedness.run(Trace())

    assert result.status == CheckStatus.PASS
    # When resolve returns None, str(None) becomes "None"
    assert result.details["inputs"]["answer"] == "None"


async def test_missing_context_in_trace() -> None:
    """Test behavior when context is not found in trace."""
    generator = MockGenerator(passed=True, reason=None)
    groundedness = Groundedness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        # No context in metadata
    )
    result = await groundedness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    # When resolve returns None with multiple=True, it returns []
    assert result.details["inputs"]["context"] == "[]"
