from typing import Any, ClassVar, override

import pytest
from giskard.agents import TemplateValidationError
from giskard.checks import CheckStatus, Interaction, LLMGenerator, LLMJudge, Trace

from .testing_utils import MockJudgeGenerator

_TRACE = Trace(interactions=[Interaction(inputs="hi", outputs="hello")])


async def test_valid_judge_prompt_runs():
    prompt = "Reply:\n{{ trace.last.outputs | fence }}"
    LLMJudge.validate_prompt(prompt, required=("trace",))

    generator = MockJudgeGenerator(passed=True, reason="ok")
    result = await LLMJudge(generator=generator, prompt=prompt).run(_TRACE)

    assert result.status == CheckStatus.PASS
    assert "hello" in (generator.calls[0][0].text or "")


@pytest.mark.parametrize(
    ("prompt", "match"),
    [
        ("{{ trace }}{{ _instr_output }}", "found: _instr_output"),
        ("{% if include_trace %}{{ trace }}{% endif %}", "found: include_trace"),
        ("{{ trace | nosuch }}", "No filter named 'nosuch'"),
        ("Is the reply polite?", "must use the variable 'trace'"),
    ],
)
def test_judge_prompt_rejected(prompt, match):
    with pytest.raises(TemplateValidationError, match=match):
        LLMJudge.validate_prompt(prompt, required=("trace",))


def test_generator_prompt_may_read_output_instructions():
    LLMGenerator.validate_prompt("Ask about {{ trace }}.\n{{ _instr_output }}")


async def test_prompt_variables_match_get_inputs():
    judge = LLMJudge(generator=MockJudgeGenerator(passed=True, reason="ok"), prompt="x")
    assert set(await judge.get_inputs(_TRACE)) == LLMJudge.prompt_variables

    generator = LLMGenerator(prompt="x")
    inputs = set(await generator.get_inputs(_TRACE)) | {"_instr_output"}
    assert inputs == LLMGenerator.prompt_variables


class _RuleJudge(LLMJudge[str, str, Trace[str, str]]):
    prompt_variables: ClassVar[frozenset[str]] = LLMJudge.prompt_variables | {"rule"}

    @override
    async def get_inputs(self, trace: Trace[str, str]) -> dict[str, Any]:
        return {**await super().get_inputs(trace), "rule": "be polite"}


def test_subclass_extends_prompt_variables():
    _RuleJudge.validate_prompt("{{ rule }}: {{ trace }}", required=("trace", "rule"))
    with pytest.raises(TemplateValidationError):
        LLMJudge.validate_prompt("{{ rule }}: {{ trace }}")
