import json
from collections.abc import Sequence
from typing import Any, cast, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Hallucination, Interaction, Trace
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import Field


class MockGenerator(BaseGenerator):
    passed: bool
    reason: str | None
    calls: list[Sequence[ChatMessage]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.calls.append(messages)
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=json.dumps(
                            {"passed": self.passed, "reason": self.reason}
                        )
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


async def test_factual_answer_passes_with_context() -> None:
    generator = MockGenerator(passed=True, reason="No hallucinated claims were found.")
    check = Hallucination(
        generator=generator,
        answer="Python was first released in 1991.",
        context="Python was first released in 1991 by Guido van Rossum.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "No hallucinated claims were found."
    assert result.details["inputs"]["answer"] == "Python was first released in 1991."
    assert "Python was first released" in result.details["inputs"]["context"]
    assert len(generator.calls) == 1


async def test_hallucinated_answer_fails_with_reason() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The answer invents a 2020 Python release date not found in context.",
    )
    check = Hallucination(
        generator=generator,
        answer="Python was created in 2020 by Ada Lovelace.",
        context="Python was first released in 1991 by Guido van Rossum.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert "invents a 2020 Python release date" in result.details["reason"]


async def test_no_context_mode_uses_empty_context() -> None:
    generator = MockGenerator(
        passed=False,
        reason="The answer contains a fabricated citation.",
    )
    check = Hallucination(
        generator=generator,
        answer="This was proven in the non-existent ZX-404 Lancet trial.",
    )

    result = await check.run(Trace())

    assert result.status == CheckStatus.FAIL
    assert result.details["inputs"]["context"] == ""
    assert "fabricated citation" in result.details["reason"]


async def test_answer_and_context_from_trace() -> None:
    generator = MockGenerator(passed=True, reason=None)
    check = Hallucination(
        generator=generator,
        answer_key="trace.interactions[0].outputs.response",
        context_key="trace.interactions[0].metadata.context",
    )
    interaction = Interaction(
        inputs={"query": "When was Python first released?"},
        outputs={"response": "Python was first released in 1991."},
        metadata={"context": "Python was first released in 1991."},
    )

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["answer"] == "Python was first released in 1991."
    context = cast(str, result.details["inputs"]["context"])
    assert "Python was first released in 1991" in context
