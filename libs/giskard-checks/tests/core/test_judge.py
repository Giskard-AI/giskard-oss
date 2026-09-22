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
    UserMessage,
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
    ("value", "expected_kind", "expected_model"),
    [
        ("openai/gpt-4o-mini", "llm", "openai/gpt-4o-mini"),
        ("llm/openai/gpt-4o-mini", "llm", "openai/gpt-4o-mini"),
        ("typesafe/jev", "som", "jev-latest"),
        ("som/typesafe/jev", "som", "jev-latest"),
    ],
)
def test_string_inference(value: str, expected_kind: str, expected_model: str):
    judge = BaseJudge.parse(value)
    assert judge.kind == expected_kind
    if expected_kind == "llm":
        assert isinstance(judge, LLMChatJudge)
        assert isinstance(judge.generator, Generator)
        assert judge.generator.model == expected_model
    else:
        assert isinstance(judge, SOMJudge)
        assert judge.model.model == expected_model


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_string_parse_is_rejected(value: str):
    with pytest.raises(ValueError, match="non-empty string"):
        BaseJudge.parse(value)


def test_dict_without_kind_infers_llm_from_generator():
    judge = BaseJudge.parse(
        {"generator": {"kind": "giskard_llm", "model": "openai/gpt-4o-mini"}}
    )
    assert isinstance(judge, LLMChatJudge)
    assert isinstance(judge.generator, Generator)
    assert judge.generator.model == "openai/gpt-4o-mini"


def test_dict_with_null_generator_and_model_infers_som():
    judge = BaseJudge.parse(
        {"generator": None, "model": {"kind": "typesafe", "model": "jev"}}
    )
    assert isinstance(judge, SOMJudge)
    assert judge.model.model == "jev-latest"


def test_top_level_som_kind_peels_judge_fields():
    judge = BaseJudge.parse({"kind": "typesafe", "model": "jev", "pass_threshold": 0.8})
    assert isinstance(judge, SOMJudge)
    assert judge.pass_threshold == 0.8
    assert judge.model.model == "jev-latest"


def test_top_level_generator_kind_wraps_llm_judge():
    judge = BaseJudge.parse({"kind": "giskard_llm", "model": "openai/gpt-4o-mini"})
    assert isinstance(judge, LLMChatJudge)
    assert isinstance(judge.generator, Generator)
    assert judge.generator.model == "openai/gpt-4o-mini"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="Kind 'nope'"):
        BaseJudge.parse({"kind": "nope"})


def test_kind_prefix_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Kind prefix must match"):
        BaseJudge.parse("llm/typesafe/jev")


def test_leaf_parse_rejects_sibling_kind():
    with pytest.raises(TypeError, match="LLMChatJudge.parse cannot produce SOMJudge"):
        LLMChatJudge.parse("som/typesafe/jev")


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
    assert "Evaluate whether the agent was polite" in evidence


async def test_som_judge_fails_closed_on_empty_evidence():
    model = RecordingSOM(model="example-v1", probability=0.99)
    judge = SOMJudge(model=model)
    prompt = (
        "{% if include_rubric | default(true) %}Is the agent polite?{% endif %}"
        "{% if include_evidence | default(true) %}{% endif %}"
    )

    verdict = await judge.judge(prompt, {})

    assert isinstance(verdict, LLMCheckResult)
    assert verdict.passed is False
    assert "empty evidence" in verdict.reason
    assert model._calls == []


async def test_som_judge_chat_message_prompt_uses_message_as_evidence():
    model = RecordingSOM(model="example-v1")
    judge = SOMJudge(model=model)
    prompt = UserMessage(content="Agent said thank you.")

    verdict = await judge.judge(prompt, {})

    assert verdict.passed is True
    messages, question = model._calls[0]
    assert messages == [prompt]
    assert "should the agent's behavior pass the check" in question
