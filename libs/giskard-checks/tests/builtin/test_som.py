"""Provider-neutral contracts for SOM judge backends."""

from collections.abc import Sequence
from typing import override
from unittest.mock import AsyncMock

import httpx
import pytest
from giskard.agents import (
    BaseSOM,
    SOMResponse,
    TemplateReference,
)
from giskard.checks import CheckStatus, Conformity, Scenario, SOMJudge
from giskard.checks.judges.base import LLMCheckResult
from giskard.llm.types import ChatMessage, Usage
from pydantic import BaseModel, PrivateAttr, ValidationError


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
    judge = SOMJudge(model=model, pass_threshold=threshold)
    prompt = TemplateReference(template_name="giskard.checks::judges/conformity.j2")
    inputs = {"rule": "Be polite", "trace": "Thank you!"}

    verdict = await judge.judge(prompt, inputs)

    assert isinstance(verdict, LLMCheckResult)
    assert verdict.passed is passed
    assert "Python decision summary: example-v1" in verdict.reason
    assert "probability instead of a generated rationale" in verdict.reason
    assert f"P(pass)={probability:.2%}" in verdict.reason
    assert f"threshold={threshold:.2%}" in verdict.reason
    assert len(model._calls) == 1
    messages, question, timeout = model._calls[0]
    assert timeout is None
    assert "JSON" not in question
    assert "schema" not in question.lower()
    assert "Thank you!" not in question
    assert "Be polite" in question
    assert "should the agent's behavior pass" not in question
    assert any("Thank you!" in (message.text or "") for message in messages)
    assert all("Be polite" not in (message.text or "") for message in messages)


async def test_timeout_override_reaches_model():
    model = MockSOM(model="example-v1")
    judge = SOMJudge(model=model, timeout=7)
    prompt = TemplateReference(template_name="giskard.checks::judges/conformity.j2")

    await judge.judge(prompt, {"rule": "Be polite", "trace": "Hello"})

    assert model._calls[0][2] == 7


async def test_custom_output_schemas_are_rejected_before_prediction():
    class CustomResult(BaseModel):
        text: str

    model = MockSOM(model="example-v1")
    judge = SOMJudge(model=model)
    with pytest.raises(ValueError, match="only LLMCheckResult"):
        await judge.judge(
            "Evaluate {{ trace }}",
            {"trace": "hi"},
            output_type=CustomResult,
        )
    assert model._calls == []


async def test_registered_som_round_trips_in_checks():
    judge = SOMJudge(
        model=MockSOM(model="example-v2", probability=0.7), pass_threshold=0.8
    )
    check = Conformity(rule="Be polite", judge=judge)
    restored_check = Conformity.model_validate_json(check.model_dump_json())

    assert isinstance(restored_check.judge, SOMJudge)
    assert isinstance(restored_check.judge.model, MockSOM)
    assert restored_check.judge.model.model == "example-v2"
    assert restored_check.judge.pass_threshold == 0.8

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
    judge = SOMJudge(model=MockSOM(model="example-v1"))

    scenario = (
        Scenario("provider-error")
        .interact("Hello", "Thank you!")
        .check(Conformity(rule="Be polite", judge=judge))
    )

    result = await scenario.run(return_exception=True)

    check = result.steps[0].results[0]
    assert check.status == CheckStatus.ERROR
    assert "HTTPStatusError" in check.details["traceback"]


async def test_invalid_prediction_remains_a_scenario_error():
    judge = SOMJudge(model=MockSOM(model="example-v1", probability=-0.1))
    scenario = (
        Scenario("invalid-prediction")
        .interact("Hello", "Thank you!")
        .check(Conformity(rule="Be polite", judge=judge))
    )

    result = await scenario.run(return_exception=True)

    check = result.steps[0].results[0]
    assert check.status == CheckStatus.ERROR
    assert "ValidationError" in check.details["traceback"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold: float):
    with pytest.raises(ValidationError):
        SOMJudge(model=MockSOM(model="example-v1"), pass_threshold=threshold)
