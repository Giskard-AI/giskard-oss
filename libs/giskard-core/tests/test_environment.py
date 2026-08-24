import sys

import pytest
from giskard.core.environment import (
    classify_environment,
    is_ci_environment,
    is_notebook_environment,
    is_truthy_env,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        (" On ", True),
    ],
)
def test_is_truthy_env(value, expected):
    assert is_truthy_env(value) is expected


def test_classify_environment_prefers_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert classify_environment() == "ci"


def test_classify_environment_detects_colab(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert classify_environment() == "colab"


def test_is_notebook_environment_includes_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", object())
    assert is_notebook_environment() is True


def test_is_ci_environment(monkeypatch):
    monkeypatch.setenv("TF_BUILD", "True")
    assert is_ci_environment() is True
