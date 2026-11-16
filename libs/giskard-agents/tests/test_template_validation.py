import re
from dataclasses import dataclass, field

import pytest
from giskard.agents.templates import (
    MessageTemplate,
    TemplateValidationError,
    strict_templates,
    validate_template,
)


@dataclass
class _Interaction:
    outputs: str = "answer"


@dataclass
class _Trace:
    interactions: list[_Interaction] = field(
        default_factory=lambda: [_Interaction("a"), _Interaction("b")]
    )

    @property
    def last(self) -> _Interaction:
        return self.interactions[-1]

    def __str__(self) -> str:
        return "\n".join(i.outputs for i in self.interactions)


VALID = [
    "{% for t in trace.interactions %}{{ loop.index }} {{ t.outputs | fence }}{% endfor %}",
    "{% set last = trace.last %}{{ last.outputs | trusted }}",
    "{% for i in range(3) %}{{ trace }}{% endfor %}",
    "{% with x = trace.last %}{{ x }}{% endwith %}",
    "{% for t in trace.interactions if t.outputs %}{{ t }}{% endfor %}",
    "{% macro m() %}x{% endmacro %}{{ m() }}{{ trace }}",
    "{% block b %}{{ trace }}{% endblock %}",
]
INVALID = [
    ("{% if trace %}a{% elseif trace.last %}b{% endif %}", "not a valid template"),
    ("{{ trace | nosuch }}", "No filter named 'nosuch'"),
    ("{% if trace is nosuchtest %}x{% endif %}", "nosuchtest"),
    ("{% if trace %}{{ trace | nosuch }}{% endif %}", "No filter named 'nosuch'"),
    ("{{ 1 if trace else trace | nosuch }}", "No filter named 'nosuch'"),
    ("{% message user %}hi{% endmessage %}{{ trace }}", "not a valid template"),
    ('{% include "x" %}{{ trace }}', "'include' tags are not allowed"),
    ('{% import "x" as y %}{{ trace }}', "'import' tags are not allowed"),
    ('{% from "x" import y %}{{ trace }}', "'from' tags are not allowed"),
    ('{% extends "x" %}', "'extends' tags are not allowed"),
    ("{{ trace }}{{ secret }}", "found: secret"),
    ("hello", "must use the variable 'trace'"),
    ("{{ trace }}{{ loop }}", "found: loop"),
    ("{{ trace }}{{ loop.index }}", "found: loop"),
    ("{% for t in trace.interactions if loop %}{{ t }}{% endfor %}", "found: loop"),
    (
        "{% for t in trace.interactions %}{{ t }}{% else %}{{ loop }}{% endfor %}",
        "found: loop",
    ),
    ("{% if 1 %}" * 100 + "{{ trace }}" + "{% endif %}" * 100, "nested too deeply"),
    ("{{" + "(" * 1990 + "1" + ")" * 1990 + "}}{{ trace }}", "nested too deeply"),
]


@pytest.mark.parametrize("source", VALID)
def test_valid_templates_pass_and_render(source):
    validate_template(source, variables={"trace"}, required={"trace"})
    MessageTemplate(role="user", content_template=source).render(trace=_Trace())


@pytest.mark.parametrize(("source", "match"), INVALID)
def test_invalid_templates_are_rejected(source, match):
    with pytest.raises(TemplateValidationError, match=re.escape(match)):
        validate_template(source, variables={"trace"}, required={"trace"})


def test_syntax_error_reports_line():
    with pytest.raises(TemplateValidationError) as error:
        validate_template("ok\n{% if %}", variables={"trace"})
    assert error.value.lineno == 2


def test_result_does_not_depend_on_policy():
    with strict_templates():
        validate_template(VALID[0], variables={"trace"}, required={"trace"})
        with pytest.raises(TemplateValidationError):
            validate_template("{{ trace | nosuch }}", variables={"trace"})


def test_required_outside_variables_is_a_caller_bug():
    with pytest.raises(ValueError, match="other") as error:
        validate_template("{{ trace }}", variables={"trace"}, required={"other"})
    assert not isinstance(error.value, TemplateValidationError)
