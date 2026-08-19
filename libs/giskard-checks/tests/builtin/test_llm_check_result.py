"""Unit tests for LLMCheckResult required non-blank reason."""

import pytest
from giskard.checks.judges.base import LLMCheckResult
from pydantic import ValidationError


@pytest.mark.parametrize(
    "payload",
    [
        {"passed": True},
        {"passed": True, "reason": None},
        {"passed": True, "reason": ""},
        {"passed": True, "reason": "   "},
        {"passed": False, "reason": "\t\n"},
    ],
    ids=["missing", "null", "empty", "whitespace", "tabs_newlines"],
)
def test_llm_check_result_rejects_missing_or_blank_reason(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _ = LLMCheckResult.model_validate(payload)


def test_llm_check_result_accepts_non_blank_reason_on_pass() -> None:
    result = LLMCheckResult.model_validate(
        {"passed": True, "reason": "Answer addresses the question."}
    )
    assert result.passed is True
    assert result.reason == "Answer addresses the question."


def test_llm_check_result_accepts_non_blank_reason_on_fail() -> None:
    result = LLMCheckResult.model_validate(
        {"passed": False, "reason": "Answer contradicts the context."}
    )
    assert result.passed is False
    assert result.reason == "Answer contradicts the context."


def test_llm_check_result_strips_surrounding_whitespace() -> None:
    result = LLMCheckResult.model_validate(
        {"passed": True, "reason": "  Grounded in context.  "}
    )
    assert result.reason == "Grounded in context."
