import json
from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Faithfulness, Interaction, Trace
from pydantic import Field


class MockGenerator(BaseGenerator):
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


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Answer faithfully represents source")
    faithfulness = Faithfulness(
        generator=generator,
        answer="The document states climate change is a serious concern.",
        source=["The document states that climate change is a serious concern requiring action."],
    )
    result = await faithfulness.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Answer faithfully represents source"

    assert len(generator.calls) == 1
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Answer distorts the source material")
    faithfulness = Faithfulness(
        generator=generator,
        answer="The document states climate change is not a concern.",
        source=["The document states that climate change is a serious concern requiring action."],
    )
    result = await faithfulness.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Answer distorts the source material"

    assert len(generator.calls) == 1


async def test_answer_and_source_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Summarize the document"},
        outputs={"response": "The document discusses climate change."},
        metadata={"source": ["The document covers climate change and its effects."]},
    )
    result = await faithfulness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None

    assert len(generator.calls) == 1
    assert "inputs" in result.details
    assert "answer" in result.details["inputs"]
    assert "source" in result.details["inputs"]
    assert result.details["inputs"]["answer"] == str(
        {"response": "The document discusses climate change."}
    )
    assert "climate change" in result.details["inputs"]["source"]


async def test_direct_answer_and_source() -> None:
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(
        generator=generator,
        answer="Direct answer",
        source=["Source 1", "Source 2"],
    )
    result = await faithfulness.run(Trace())

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["answer"] == "Direct answer"
    assert "Source 1" in result.details["inputs"]["source"]
    assert "Source 2" in result.details["inputs"]["source"]


async def test_direct_answer_and_single_string_source() -> None:
    """Test that source can be a single string instead of a list."""
    generator = MockGenerator(passed=True, reason="Answer faithfully represents source")
    faithfulness = Faithfulness(
        generator=generator,
        answer="Climate change is a serious concern.",
        source="The document states that climate change is a serious concern requiring immediate action.",
    )
    result = await faithfulness.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Climate change is a serious concern."
    assert (
        result.details["inputs"]["source"]
        == "The document states that climate change is a serious concern requiring immediate action."
    )
    assert len(generator.calls) == 1


async def test_answer_priority_over_trace() -> None:
    """Test that direct answer takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(
        generator=generator,
        answer="Direct answer takes priority",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Trace answer"},
    )
    result = await faithfulness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Direct answer takes priority"


async def test_source_priority_over_trace() -> None:
    """Test that direct source takes priority over trace extraction."""
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(
        generator=generator,
        source=["Direct source"],
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"source": ["Trace source"]},
    )
    result = await faithfulness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["source"] == "['Direct source']"


async def test_missing_answer_in_trace() -> None:
    """Test behavior when answer is not found in trace."""
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(generator=generator)
    result = await faithfulness.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "No match for key: trace.last.outputs"


async def test_missing_source_in_trace() -> None:
    """Test behavior when source is not found in trace."""
    generator = MockGenerator(passed=True, reason=None)
    faithfulness = Faithfulness(generator=generator)
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
    )
    result = await faithfulness.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert (
        result.details["inputs"]["source"]
        == "No match for key: trace.last.metadata.source"
    )