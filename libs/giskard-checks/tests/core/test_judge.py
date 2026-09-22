"""Unit tests for discriminated judge backends."""

from collections.abc import Sequence
from typing import Any, override

import pytest
from giskard.agents import (
    BaseGenerator,
    BaseSOM,
    GenerationParams,
    Generator,
    SOMResponse,
    TemplateReference,
)
from giskard.checks import (
    BaseJudge,
    Conformity,
    Interaction,
    LLMChatJudge,
    SOMJudge,
    Trace,
)
from giskard.checks.judges.base import LLMCheckResult
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
)
from pydantic import PrivateAttr


@BaseSOM.register("checks_judge_test_som")
class RecordingSOM(BaseSOM):
    probability: float = 0.9
    _calls: list[tuple[Sequence[ChatMessage], str]] = PrivateAttr(default_factory=list)

    @override
    async def predict(
        self,
        messages: Sequence[ChatMessage],
        question: str,
        *,
        timeout: float | int | None = None,
    ) -> SOMResponse:
        self._calls.append((messages, question))
        return SOMResponse(model=self.model, probability=self.probability)


@BaseGenerator.register("checks_judge_test_llm")
class RecordingLLM(BaseGenerator):
    _messages: list[Sequence[ChatMessage]] = PrivateAttr(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self._messages.append(messages)
        verdict = LLMCheckResult(passed=True, reason="ok")
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content=verdict.model_dump_json()),
                    finish_reason="stop",
                )
            ]
        )


def test_judge_kinds():
    assert BaseJudge.kinds() == frozenset({"llm", "som"})


@pytest.mark.parametrize(
    ("value", "expected_kind"),
    [
        ("openai/gpt-4o-mini", "llm"),
        ("llm/openai/gpt-4o-mini", "llm"),
        ("typesafe/jev", "som"),
        ("som/typesafe/jev", "som"),
    ],
)
def test_string_inference(value: str, expected_kind: str):
    judge = BaseJudge.parse(value)
    assert judge.kind == expected_kind


def test_dict_without_kind_infers_llm_from_generator():
    judge = BaseJudge.parse(
        {"generator": {"kind": "giskard_llm", "model": "openai/gpt-4o-mini"}}
    )
    assert isinstance(judge, LLMChatJudge)
    assert isinstance(judge.generator, Generator)
    assert judge.generator.model == "openai/gpt-4o-mini"


def test_round_trip_preserves_kind():
    judge = SOMJudge(model=RecordingSOM(model="example-v1"), pass_threshold=0.8)
    restored = BaseJudge.model_validate(judge.model_dump())
    assert isinstance(restored, SOMJudge)
    assert restored.pass_threshold == 0.8
    assert restored.model.model == "example-v1"


async def test_llm_judge_keeps_evidence_and_output_instructions():
    generator = RecordingLLM()
    judge = LLMChatJudge(generator=generator)
    check = Conformity(rule="Be polite", judge=judge)
    trace = Trace(interactions=[Interaction(inputs="Hello", outputs="Thank you!")])

    result = await check.run(trace)

    assert result.status.value == "pass"
    assert len(generator._messages) == 1
    rendered = generator._messages[0][0].text or ""
    assert "Be polite" in rendered
    assert "Thank you!" in rendered
    assert "JSON" in rendered or "schema" in rendered.lower()


async def test_som_judge_splits_question_and_evidence():
    model = RecordingSOM(model="example-v1")
    judge = SOMJudge(model=model)
    trace = Trace(interactions=[Interaction(inputs="Hello", outputs="Thank you!")])

    verdict = await judge.judge(
        TemplateReference(template_name="giskard.checks::judges/conformity.j2"),
        {"rule": "Be polite", "trace": trace},
    )

    assert isinstance(verdict, LLMCheckResult)
    assert verdict.passed is True
    messages, question = model._calls[0]
    assert "Be polite" in question
    assert "JSON" not in question
    assert "schema" not in question.lower()
    evidence = messages[0].text or ""
    assert "Thank you!" in evidence
    assert "Be polite" not in evidence


async def test_som_judge_falls_back_when_prompt_has_no_rubric_gates():
    model = RecordingSOM(model="example-v1")
    judge = SOMJudge(model=model)

    verdict = await judge.judge(
        "Evaluate whether the agent was polite.\n\n{{ trace | fence }}",
        {"trace": "Thank you!"},
    )

    assert isinstance(verdict, LLMCheckResult)
    assert verdict.passed is True
    messages, question = model._calls[0]
    assert "Using the rubric and evidence in the evaluation prompt" in question
    assert "should the agent's behavior pass the check" in question
    evidence = messages[0].text or ""
    assert "Thank you!" in evidence
    assert "Evaluate whether the agent was polite" not in evidence
