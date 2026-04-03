import json
from typing import Any, cast, override

import pytest
from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import Check, CheckStatus, Interaction, LLMJudge, Trace
from pydantic import Field, ValidationError


@BaseGenerator.register("mock")
class MockGenerator(BaseGenerator):
    """Mock generator that returns predictable pass/fail and reason."""

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


def serialization_roundtrip[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    judge: LLMJudge[InputType, OutputType, TraceType],
) -> LLMJudge[InputType, OutputType, TraceType]:
    check = Check.model_validate(judge.model_dump())
    assert isinstance(check, LLMJudge)
    return cast(LLMJudge[InputType, OutputType, TraceType], check)


async def test_custom_generator_preserved_after_serialization_roundtrip() -> None:
    """Custom generator is preserved across model_dump/model_validate (fixes #2292)."""
    generator = MockGenerator(passed=True, reason="Preserved reason")
    judge = LLMJudge(generator=generator, prompt="Evaluate.")
    roundtrip_judge = serialization_roundtrip(judge)

    # Generator is preserved by roundtrip (no manual re-attachment needed)
    assert roundtrip_judge.generator is not None
    assert isinstance(roundtrip_judge.generator, MockGenerator)
    assert roundtrip_judge.generator.passed is True
    assert roundtrip_judge.generator.reason == "Preserved reason"

    result = await roundtrip_judge.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Preserved reason"
    assert len(roundtrip_judge.generator.calls) == 1


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Looks good")
    judge = LLMJudge(generator=generator, prompt="Evaluate the answer.")
    result = await judge.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Looks good"

    assert len(generator.calls) == 1
    assert generator.calls[0] == [Message(role="user", content="Evaluate the answer.")]

    roundtrip_judge = serialization_roundtrip(judge)
    result = await roundtrip_judge.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Looks good"
    assert isinstance(roundtrip_judge.generator, MockGenerator)
    # Generator state (including calls) is preserved by roundtrip; one more call from this run
    assert len(roundtrip_judge.generator.calls) == 2
    assert roundtrip_judge.generator.calls[-1] == [
        Message(role="user", content="Evaluate the answer.")
    ]


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Looks bad")
    judge = LLMJudge(generator=generator, prompt="Evaluate the answer.")
    result = await judge.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Looks bad"

    assert len(generator.calls) == 1
    assert generator.calls[0] == [Message(role="user", content="Evaluate the answer.")]

    roundtrip_judge = serialization_roundtrip(judge)
    result = await roundtrip_judge.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Looks bad"
    assert isinstance(roundtrip_judge.generator, MockGenerator)
    assert len(roundtrip_judge.generator.calls) == 2
    assert roundtrip_judge.generator.calls[-1] == [
        Message(role="user", content="Evaluate the answer.")
    ]


async def test_run_handle_template_reference() -> None:
    generator = MockGenerator(passed=True, reason=None)
    judge = LLMJudge(
        generator=generator,
        prompt="Evaluate the answer: {{ trace.interactions[-1].outputs.response }}",
    )
    result = await judge.run(
        Trace(
            interactions=[
                Interaction(inputs={"response": "Hello"}, outputs={"response": "Hello"})
            ]
        )
    )

    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None

    assert len(generator.calls) == 1
    assert generator.calls[0] == [
        Message(role="user", content="Evaluate the answer: Hello")
    ]

    roundtrip_judge = serialization_roundtrip(judge)
    result = await roundtrip_judge.run(
        Trace(
            interactions=[
                Interaction(inputs={"response": "Hello"}, outputs={"response": "Hello"})
            ]
        )
    )
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] is None
    assert isinstance(roundtrip_judge.generator, MockGenerator)
    assert len(roundtrip_judge.generator.calls) == 2
    assert roundtrip_judge.generator.calls[-1] == [
        Message(role="user", content="Evaluate the answer: Hello")
    ]


async def test_validate_no_prompt_or_path() -> None:
    generator = MockGenerator(passed=True, reason=None)

    with pytest.raises(
        ValidationError, match="Either 'prompt' or 'prompt_path' must be provided"
    ):
        _ = LLMJudge(generator=generator)


async def test_validate_both_prompt_or_path() -> None:
    generator = MockGenerator(passed=True, reason=None)

    with pytest.raises(
        ValidationError,
        match="Cannot provide both 'prompt' and 'prompt_path' - choose one",
    ):
        _ = LLMJudge(
            generator=generator,
            prompt="Evaluate the answer.",
            prompt_path="prompts/judge_prompt.j2",
        )
