"""``target_key`` is the canonical name for the subject of every check.

The field naming the value under test (the "subject") is called ``target_key``
on every check, matching the Giskard Hub. Each check additionally accepts its
domain-meaningful spelling and its legacy spelling as read-only validation
aliases, so persisted suites and existing user code keep loading. Serialization
is one-way: ``model_dump()`` always emits ``target_key``.
"""

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

# (check class, extra kwargs needed to construct it, aliases besides target_key)
CASES: list[tuple[type[Check[Any, Any, Any]], dict[str, Any], tuple[str, ...]]] = [
    (Equals, {"expected_value": 5}, ("actual_key", "key")),
    (StringMatching, {"keyword": "x"}, ("text_key",)),
    (RegexMatching, {"pattern": "x"}, ("text_key",)),
    (JsonValid, {}, ("value_key", "key")),
    (Readability, {}, ("text_key", "key")),
    (SemanticSimilarity, {}, ("answer_key", "actual_answer_key")),
    (Toxicity, {}, ("output_key",)),
    (Groundedness, {}, ("answer_key",)),
    (Contradiction, {}, ("answer_key",)),
    (AnswerRelevance, {}, ("answer_key",)),
]

IDS = [case[0].__name__ for case in CASES]


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_target_key_is_the_canonical_field(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """``target_key`` works as a python kwarg and is readable as an attribute."""
    # ``cls`` is typed as the base ``Check``, which declares no ``target_key``;
    # that the concrete subclasses do is exactly what this asserts.
    check = cls(target_key=_SENTINEL, **kwargs)  # pyright: ignore[reportCallIssue]

    assert check.target_key == _SENTINEL  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_every_alias_is_accepted_on_validation(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """Legacy and domain spellings still populate ``target_key``."""
    for alias in aliases:
        check = cls.model_validate({**kwargs, alias: _SENTINEL})
        assert check.target_key == _SENTINEL, alias  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_dump_emits_target_key_only(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """Serialization is canonical: ``target_key`` in, old names out."""
    for alias in aliases:
        payload = cls.model_validate({**kwargs, alias: _SENTINEL}).model_dump()
        assert payload["target_key"] == _SENTINEL
        for old in aliases:
            if old != "target_key":
                assert old not in payload, f"{old} leaked into the dump"


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_round_trips_through_dump(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """``model_validate(model_dump())`` reconstructs an equal check."""
    check = cls(target_key=_SENTINEL, **kwargs)  # pyright: ignore[reportCallIssue]

    assert cls.model_validate(check.model_dump()) == check
    assert cls.model_validate_json(check.model_dump_json()) == check


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_unknown_key_still_forbidden(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """Aliasing must not weaken ``extra="forbid"``."""
    with pytest.raises(ValidationError) as exc_info:
        cls.model_validate({**kwargs, "definitely_not_a_key": _SENTINEL})

    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


@pytest.mark.parametrize(("cls", "kwargs", "aliases"), CASES, ids=IDS)
def test_default_target_key_unchanged(
    cls: type[Check[Any, Any, Any]],
    kwargs: dict[str, Any],
    aliases: tuple[str, ...],
) -> None:
    """The rename must not move any default."""
    assert cls(**kwargs).target_key == "trace.last.outputs"  # pyright: ignore[reportAttributeAccessIssue]
