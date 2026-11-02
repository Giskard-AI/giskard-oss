"""Tests for shared integration helpers."""

import math

import pytest
from giskard.scan.integrations._shared import is_valid_score


@pytest.mark.parametrize(
    "score",
    [0.0, 1.0, 0.5, 0.999, 1, 0],
    ids=["zero_float", "one_float", "half_float", "near_one", "one_int", "zero_int"],
)
def test_is_valid_score_valid(score: object) -> None:
    assert is_valid_score(score) is True


@pytest.mark.parametrize(
    "score",
    [
        None,
        True,
        False,
        math.nan,
        math.inf,
        -math.inf,
        -0.0001,
        -1.0,
        1.0001,
        2.0,
        "0.5",
        "invalid",
        [],
        {},
        object(),
    ],
    ids=[
        "none",
        "bool_true",
        "bool_false",
        "nan",
        "inf",
        "neg_inf",
        "negative_epsilon",
        "negative_one",
        "above_one_epsilon",
        "two",
        "string_number",
        "string_text",
        "list",
        "dict",
        "object",
    ],
)
def test_is_valid_score_invalid(score: object) -> None:
    assert is_valid_score(score) is False
