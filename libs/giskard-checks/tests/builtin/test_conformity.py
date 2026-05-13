import json
from collections.abc import Sequence
from typing import Any, override

from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Conformity, Interaction, Trace
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
)
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


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Rule is followed")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Rule is followed"

    assert len(generator.calls) == 1
    # The prompt comes from the template file, so we check that the call was made
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Rule is violated")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Rule is violated"

    assert len(generator.calls) == 1


async def test_trace_in_result_details_inputs() -> None:
    generator = MockGenerator(passed=True, reason=None)
    conformity = Conformity(generator=generator, rule="Test rule")
    interaction = Interaction(
        inputs={"query": "What is AI?"}, outputs={"response": "AI is..."}
    )
    trace = Trace(interactions=[interaction])
    result = await conformity.run(trace)

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["rule"] == "Test rule"
    assert "trace" in result.details["inputs"]

    stored_trace = result.details["inputs"]["trace"]
    assert stored_trace is trace


async def test_empty_trace_in_result_details_inputs() -> None:
    generator = MockGenerator(passed=True, reason=None)
    conformity = Conformity(generator=generator, rule="Test rule")
    trace = Trace()
    result = await conformity.run(trace)

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["rule"] == "Test rule"
    stored_trace = result.details["inputs"]["trace"]
    assert stored_trace is trace
