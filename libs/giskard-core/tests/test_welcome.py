import types

import pytest
from giskard.core import environment, welcome


@pytest.fixture(autouse=True)
def reset_welcome_state():
    welcome._shown = False
    yield
    welcome._shown = False


@pytest.mark.parametrize(
    ("env", "stderr_tty", "notebook_modules", "expected"),
    [
        ({}, True, [], True),
        ({"GISKARD_HIDE_WELCOME": "1"}, True, [], False),
        ({"GISKARD_HIDE_WELCOME": ""}, True, [], True),
        ({"GISKARD_HIDE_WELCOME": "maybe"}, True, [], True),
        ({"CI": "true"}, True, [], False),
        ({"TF_BUILD": "True"}, True, [], False),
        ({"PYTEST_VERSION": "8.0.0"}, True, [], False),
        ({}, False, [], False),
        ({}, False, ["IPython"], True),
        ({}, False, ["google.colab"], True),
    ],
)
def test_should_show_welcome(
    monkeypatch,
    env,
    stderr_tty,
    notebook_modules,
    expected,
):
    monkeypatch.delenv("GISKARD_HIDE_WELCOME", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("TF_BUILD", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    for module_name in ("IPython", "google.colab"):
        monkeypatch.delitem(environment.sys.modules, module_name, raising=False)
    for module_name in notebook_modules:
        monkeypatch.setitem(
            environment.sys.modules, module_name, types.ModuleType(module_name)
        )

    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: stderr_tty)

    assert welcome._should_show_welcome() is expected


def test_maybe_show_welcome_prints_to_stderr(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: True)

    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert "Thank you for using Giskard open-source!" in captured.err
    assert captured.out == ""
    assert welcome._shown is True


def test_maybe_show_welcome_prints_once(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: True)

    welcome.maybe_show_welcome()
    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert captured.err.count("Thank you for using Giskard open-source!") == 1


def test_maybe_show_welcome_does_not_raise_on_invalid_hide_welcome(monkeypatch, capsys):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", "maybe")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: True)

    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert "Thank you for using Giskard open-source!" in captured.err
    assert welcome._shown is True


def test_maybe_show_welcome_swallows_settings_errors(monkeypatch):
    def boom() -> object:
        raise RuntimeError("settings failed")

    monkeypatch.setattr(welcome, "get_settings", boom)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: True)

    welcome.maybe_show_welcome()
    assert welcome._shown is False
