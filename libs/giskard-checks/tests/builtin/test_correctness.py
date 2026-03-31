import json
from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, Correctness, Interaction, Trace
from pydantic import Field


class LLMTrace(Trace[str, str], frozen=True):
    def _repr_prompt_(self) -> str:
        return "\n".join(
            [
                f"[user]: {interaction.inputs}\n[assistant]: {interaction.outputs}"
                for interaction in self.interactions
            ]
        )


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
    generator = MockGenerator(passed=True, reason="Matches reference")
    correctness = Correctness(
        generator=generator,
        description="Q&A bot",
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
        description="Q&A bot",
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
    message = generator.calls[0][0]
    assert isinstance(message.content, str)
    assert (
        "<AGENT DESCRIPTION>\nFactual assistant\n</AGENT DESCRIPTION>"
        in message.content
    )
    assert (
        f"<CONVERSATION>\n{trace._repr_prompt_()}\n</CONVERSATION>" in message.content
    )
    assert "<AGENT ANSWER>\nParis.\n</AGENT ANSWER>" in message.content
    assert (
        "<REFERENCE ANSWER>\nParis is the capital of France.\n</REFERENCE ANSWER>"
        in message.content
    )
