"""Provider-neutral contracts for adapting SOM probabilities to check verdicts."""

from collections.abc import Sequence
from typing import Any, override
from unittest.mock import AsyncMock

import httpx
import pytest
from giskard.agents import (
    BaseGenerator,
    BaseSOM,
    GenerationParams,
    SOMResponse,
    tool,
)
from giskard.agents.generators.middleware import CompletionMiddleware, NextFn
from giskard.checks import CheckStatus, Conformity, Scenario, SOMJudgeGenerator
from giskard.checks.judges.base import LLMCheckResult
from giskard.llm.types import ChatMessage, ChatMessageParam, CompletionResponse, Usage
from pydantic import BaseModel, PrivateAttr, ValidationError

MESSAGES: list[ChatMessageParam] = [
    {"role": "user", "content": "Rubric: pass if polite. Answer: Thank you!"}
]
JUDGE_PARAMS = GenerationParams(response_format=LLMCheckResult)


@BaseSOM.register("checks_test_som")
class MockSOM(BaseSOM):
    """A model without any provider-specific transport or configuration."""

    probability: float = 0.9
    usage: Usage | None = None
    _calls: list[tuple[Sequence[ChatMessage], str, float | int | None]] = PrivateAttr(
        default_factory=list
    )

    @override
    async def predict(
        self,
        messages: Sequence[ChatMessage],
        question: str,
        *,
        timeout: float | int | None = None,
    ) -> SOMResponse:
        self._calls.append((messages, question, timeout))
        return SOMResponse(
            model=self.model, probability=self.probability, usage=self.usage
        )


@CompletionMiddleware.register("checks_test_som_recorder")
class RecordingMiddleware(CompletionMiddleware):
    _metadata: dict[str, Any] | None = PrivateAttr(default=None)
    _response: CompletionResponse | None = PrivateAttr(default=None)

    @override
    async def call(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams | None,
        metadata: dict[str, Any] | None,
        next_fn: NextFn,
    ) -> CompletionResponse:
        self._metadata = metadata
        self._response = await next_fn(messages, params, metadata)
        return self._response


@pytest.mark.parametrize(
    "probability, threshold, passed",
    [
        (0, 0.5, False),
        (0.49, 0.5, False),
        (0.5, 0.5, True),
        (1, 0.5, True),
        (0.7, 0.8, False),
    ],
)
async def test_probability_becomes_a_verdict(
    probability: float, threshold: float, passed: bool
):
    model = MockSOM(model="example-v1", probability=probability)
    judge = SOMJudgeGenerator(model=model, pass_threshold=threshold)

    completion = await judge.complete(MESSAGES, JUDGE_PARAMS)
    text = completion.choices[0].message.text
    assert text is not None
    verdict = LLMCheckResult.model_validate_json(text)

    assert verdict.passed is passed
    assert "Python decision summary: example-v1" in verdict.reason
    assert "probability instead of a generated rationale" in verdict.reason
    assert f"P(pass)={probability:.2%}" in verdict.reason
    assert f"threshold={threshold:.2%}" in verdict.reason
    assert completion.model == "example-v1"
    assert completion.choices[0].finish_reason == "stop"
    assert len(model._calls) == 1
    messages, question, timeout = model._calls[0]
    assert [message.text for message in messages] == [MESSAGES[0].get("content")]
    assert "should the agent's behavior pass the check?" in question
    assert timeout is None


@pytest.mark.parametrize(
    "usage", [None, Usage(input_tokens=300, output_tokens=20, total_tokens=320)]
)
async def test_usage_and_metadata_reach_completion_middleware(usage: Usage | None):
    middleware = RecordingMiddleware()
    model = MockSOM(model="example-v1", usage=usage)
    judge = SOMJudgeGenerator(model=model, middlewares=[middleware])
    metadata = {"trace_id": "judge-trace"}

    completion = await judge.complete(MESSAGES, JUDGE_PARAMS, metadata)

    assert completion.usage == usage
    assert middleware._metadata == metadata
    assert middleware._response is completion
    assert middleware._response is not None
    assert middleware._response.model == "example-v1"


async def test_timeout_override_reaches_model():
    model = MockSOM(model="example-v1")
    judge = SOMJudgeGenerator(model=model, params=GenerationParams(timeout=30))

    await judge.complete(
        MESSAGES, GenerationParams(response_format=LLMCheckResult, timeout=7)
    )

    assert model._calls[0][2] == 7


async def test_generation_and_custom_schemas_are_rejected_before_prediction():
    class CustomResult(BaseModel):
        text: str

    model = MockSOM(model="example-v1")
    judge = SOMJudgeGenerator(model=model)
    for params in [None, GenerationParams(response_format=CustomResult)]:
        with pytest.raises(ValueError, match="only LLMCheckResult"):
            await judge.complete(MESSAGES, params)
    assert model._calls == []


async def test_tools_are_rejected_before_prediction():
    @tool
    def example_tool() -> str:
        """Return an example."""
        return "example"

    model = MockSOM(model="example-v1")
    with pytest.raises(ValueError, match="without tools"):
        await SOMJudgeGenerator(model=model).complete(
            MESSAGES,
            GenerationParams(response_format=LLMCheckResult, tools=[example_tool]),
        )
    assert model._calls == []


async def test_registered_som_round_trips_in_generators_and_checks():
    judge = SOMJudgeGenerator(
        model=MockSOM(model="example-v2", probability=0.7), pass_threshold=0.8
    )
    restored = BaseGenerator.model_validate_json(judge.model_dump_json())

    assert isinstance(restored, SOMJudgeGenerator)
    assert isinstance(restored.model, MockSOM)
    assert restored.model.model == "example-v2"
    assert restored.pass_threshold == 0.8

    check = Conformity(rule="Be polite", generator=restored)
    restored_check = Conformity.model_validate_json(check.model_dump_json())
    assert isinstance(restored_check.generator, SOMJudgeGenerator)
    assert isinstance(restored_check.generator.model, MockSOM)
    scenario = (
        Scenario("restored-som").interact("Hello", "Thank you!").check(restored_check)
    )

    result = await scenario.run(return_exception=True)

    assert result.steps[0].results[0].status == CheckStatus.FAIL


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_failure_remains_a_scenario_error(
    monkeypatch: pytest.MonkeyPatch, status: int
):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://som.example/predict")
    )
    error = httpx.HTTPStatusError(
        "Provider failure", request=response.request, response=response
    )
    monkeypatch.setattr(MockSOM, "predict", AsyncMock(side_effect=error))
    judge = SOMJudgeGenerator(model=MockSOM(model="example-v1"))
    with pytest.raises(httpx.HTTPStatusError) as provider_error:
        await judge.complete(MESSAGES, JUDGE_PARAMS)
    assert provider_error.value.response.status_code == status

    scenario = (
        Scenario("provider-error")
        .interact("Hello", "Thank you!")
        .check(Conformity(rule="Be polite", generator=judge))
    )

    result = await scenario.run(return_exception=True)

    check = result.steps[0].results[0]
    assert check.status == CheckStatus.ERROR
    assert "HTTPStatusError" in check.details["traceback"]


async def test_invalid_prediction_remains_a_scenario_error():
    judge = SOMJudgeGenerator(model=MockSOM(model="example-v1", probability=-0.1))
    with pytest.raises(ValidationError):
        await judge.complete(MESSAGES, JUDGE_PARAMS)
    scenario = (
        Scenario("invalid-prediction")
        .interact("Hello", "Thank you!")
        .check(Conformity(rule="Be polite", generator=judge))
    )

    result = await scenario.run(return_exception=True)

    check = result.steps[0].results[0]
    assert check.status == CheckStatus.ERROR
    assert "ValidationError" in check.details["traceback"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold: float):
    with pytest.raises(ValidationError):
        SOMJudgeGenerator(model=MockSOM(model="example-v1"), pass_threshold=threshold)
