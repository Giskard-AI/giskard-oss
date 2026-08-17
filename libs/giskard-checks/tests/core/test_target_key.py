"""``target_key`` is the canonical subject field on every check."""

from typing import Any

import pytest
from giskard.checks.builtin.comparison import Equals
from giskard.checks.builtin.json_valid import JsonValid
from giskard.checks.builtin.nlp_metrics import Readability
from giskard.checks.builtin.semantic_similarity import SemanticSimilarity
from giskard.checks.builtin.text_matching import RegexMatching, StringMatching
from giskard.checks.core.check import Check
from giskard.checks.judges.answer_relevance import AnswerRelevance
from giskard.checks.judges.contradiction import Contradiction
from giskard.checks.judges.groundedness import Groundedness
from giskard.checks.judges.toxicity import Toxicity
from pydantic import ValidationError

pytest.importorskip("textstat", reason="Readability requires the textstat extra")

_SENTINEL = "trace.last.metadata.subject_under_test"

CASES: list[tuple[type[Check[Any, Any, Any]], dict[str, Any]]] = [
    (Equals, {"expected_value": 5}),
    (StringMatching, {"keyword": "x"}),
    (RegexMatching, {"pattern": "x"}),
    (JsonValid, {}),
    (Readability, {}),
    (SemanticSimilarity, {}),
    (Toxicity, {}),
    (Groundedness, {}),
    (Contradiction, {}),
    (AnswerRelevance, {}),
]

IDS = [case[0].__name__ for case in CASES]


@pytest.mark.parametrize(("cls", "kwargs"), CASES, ids=IDS)
def test_target_key_is_the_subject_field(
    cls: type[Check[Any, Any, Any]], kwargs: dict[str, Any]
) -> None:
    check = cls(target_key=_SENTINEL, **kwargs)  # pyright: ignore[reportCallIssue]

    assert check.target_key == _SENTINEL  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("cls", "kwargs"), CASES, ids=IDS)
def test_target_key_round_trips(
    cls: type[Check[Any, Any, Any]], kwargs: dict[str, Any]
) -> None:
    check = cls(target_key=_SENTINEL, **kwargs)  # pyright: ignore[reportCallIssue]

    assert cls.model_validate(check.model_dump()) == check
    assert cls.model_validate_json(check.model_dump_json()) == check


@pytest.mark.parametrize(("cls", "kwargs"), CASES, ids=IDS)
def test_old_subject_key_names_are_forbidden(
    cls: type[Check[Any, Any, Any]], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        cls.model_validate({**kwargs, "key": _SENTINEL})

    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


@pytest.mark.parametrize(("cls", "kwargs"), CASES, ids=IDS)
def test_default_target_key_unchanged(
    cls: type[Check[Any, Any, Any]], kwargs: dict[str, Any]
) -> None:
    assert cls(**kwargs).target_key == "trace.last.outputs"  # pyright: ignore[reportAttributeAccessIssue]
