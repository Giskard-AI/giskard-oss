"""Tests for shared integration helpers."""

import math

import numpy as np
import pytest
from giskard.scan.integrations._shared import is_valid_score


@pytest.mark.parametrize(
    "score",
    [
        0.0,
        1.0,
        0.5,
        0.999,
        1,
        0,
        np.float32(1.0),
        np.float64(0.5),
        np.float16(0.25),
        np.int32(1),
        np.int64(0),
    ],
    ids=[
        "zero_float",
        "one_float",
        "half_float",
        "near_one",
        "one_int",
        "zero_int",
        "np_float32",
        "np_float64",
        "np_float16",
        "np_int32",
        "np_int64",
    ],
)
def test_is_valid_score_valid(score: object) -> None:
    assert is_valid_score(score) is True


@pytest.mark.parametrize(
    "score",
    [
        None,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        np.float32(-0.5),
        np.float32(1.5),
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
        "np_bool_true",
        "np_bool_false",
        "np_nan",
        "np_inf",
        "np_neg_inf",
        "np_float32_neg",
        "np_float32_over",
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
