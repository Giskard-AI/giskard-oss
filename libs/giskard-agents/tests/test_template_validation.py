import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import pytest
from giskard.agents.templates import (
    MessageTemplate,
    TemplateValidationError,
    strict_templates,
    validate_template,
    validation,
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
    "{{ trace.interactions | map(attribute='outputs') | map('upper') | join }}",
    "{{ trace.interactions | selectattr('outputs', 'string') | list | length }}",
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
    ("{{ trace.interactions | map('nosuch') | list }}", "No filter named 'nosuch'"),
    ("{{ trace.interactions | select('nosuch') | list }}", "No test named 'nosuch'"),
    (
        "{{ trace.interactions | rejectattr('outputs', 'nosuch') | list }}",
        "No test named 'nosuch'",
    ),
    ("{{ 1" + "0" * 5000 + " }}{{ trace }}", "not a valid template"),
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


# Jinja folds constant expressions while compiling, so validating these used to
# build gigabyte strings or huge integers before rendering was ever attempted.
# Each is valid, and well under the 4000-character limit for judge templates.
EXPENSIVE_CONSTANTS = [
    "{{ 'a' * 300000000 }}",
    "{{ ('a' + 'b') * 300000000 }}",
    "{{ ('a' ~ 'b') * 300000000 }}",
    "{{ '%s' % ('a' * 300000000) }}",
    "{{ 10 ** 100000000 }}",
    "{{ 2 ** 30000000 // 3 }}",
    "{{ 9 ** 9 ** 9 }}",
    "{{ -(10 ** 100000000) }}",
    "{{ 'x' | center(1000000000) }}",
    "{{ 'x' | center(10 ** 9) }}",
    "{{ 'a\nb' | indent(1000000000) }}",
    "{{ ('a ' * 50000000) | wordwrap(1) }}",
    "{{ ('a' * 10000) | replace('a', 'a' * 100000) }}",
    "{{ ('a' * 100000) | join('b' * 10000) }}",
    "{{ ('a' * 100000000) | batch(1) | list }}",
    "{{ 'a' | truncate(10 ** 9) | center(10 ** 9) }}",
    "{% if 'a' * 300000000 is string %}x{% endif %}",
    "{% set x = 'a' * 300000000 %}",
    "{% for i in ['a' * 300000000] %}{{ i }}{% endfor %}",
    "{% block b %}{{ 'a' * 300000000 }}{% endblock %}",
    "{% macro m() %}{{ 'x' | center(1000000000) }}{% endmacro %}",
    "{% autoescape 'a' * 300000000 %}x{% endautoescape %}",
    "{% filter center(1000000000) %}x{% endfilter %}",
]

_VALIDATE_UNDER_MEMORY_CAP = """
import resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
from giskard.agents.templates import TemplateValidationError, validate_template
start = time.perf_counter()
try:
    validate_template(sys.argv[1], variables={"trace"}, required={"trace"})
except TemplateValidationError:
    pass
print(time.perf_counter() - start)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="needs RLIMIT_AS")
@pytest.mark.parametrize("source", EXPENSIVE_CONSTANTS)
def test_constant_expressions_are_not_evaluated(source):
    # A subprocess with 1 GiB of address space, so a regression fails the
    # test (with a MemoryError, or by being slow) instead of the machine.
    result = subprocess.run(
        [sys.executable, "-c", _VALIDATE_UNDER_MEMORY_CAP, "{{ trace }}" + source],
        env={**os.environ, "DO_NOT_TRACK": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout) < 0.5


def test_constant_expressions_still_validate_and_render():
    source = "{{ trace }}{{ 'ab' * 2 ~ -1 }}{{ 'x' | center(5) }}{{ 2 ** 3 // 3 % 2 }}"
    validate_template(source, variables={"trace"}, required={"trace"})
    rendered = MessageTemplate(role="user", content_template=source).render(
        trace=_Trace()
    )
    assert rendered.content == "a\nbabab-1  x  0"


def test_unknown_filter_with_constant_arguments_is_rejected():
    with pytest.raises(TemplateValidationError, match="No filter named 'nosuch'"):
        validate_template("{{ trace }}{{ 'x' | nosuch(2 * 3) }}", variables={"trace"})


@pytest.mark.parametrize("error", [MemoryError, OverflowError])
def test_resource_errors_are_validation_errors(monkeypatch, error):
    def compile(*args, **kwargs):
        raise error()

    monkeypatch.setattr(validation._validation_env, "compile", compile)
    with pytest.raises(TemplateValidationError, match="too large"):
        validate_template("{{ trace }}", variables={"trace"})
