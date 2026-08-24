import pytest
from giskard.core import environment, welcome


@pytest.fixture(autouse=True)
def reset_welcome_state():
    welcome._shown = False
    yield
    welcome._shown = False


def test_import_does_not_print_welcome(capsys):
    capsys.readouterr()
    assert welcome._shown is False


@pytest.mark.parametrize(
    ("env", "stderr_tty", "notebook_modules"),
    [
        ({}, True, []),
        ({"GISKARD_HIDE_WELCOME": "1"}, True, []),
        ({"CI": "true"}, True, []),
        ({"TF_BUILD": "True"}, True, []),
        ({"PYTEST_VERSION": "8.0.0"}, True, []),
        ({}, False, []),
        ({}, False, ["IPython"]),
        ({}, False, ["google.colab"]),
    ],
)
def test_maybe_show_welcome_gates(
    monkeypatch,
    capsys,
    env,
    stderr_tty,
    notebook_modules,
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
        environment.sys.modules[module_name] = object()

    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: stderr_tty)

    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    should_show = (
        env.get("GISKARD_HIDE_WELCOME") is None
        and env.get("CI") is None
        and env.get("TF_BUILD") is None
        and env.get("PYTEST_VERSION") is None
        and (stderr_tty or notebook_modules)
    )

    if should_show:
        assert "Thank you for using Giskard open-source!" in captured.err
        assert captured.out == ""
        assert welcome._shown is True
    else:
        assert captured.err == ""
        assert captured.out == ""
        assert welcome._shown is False


def test_maybe_show_welcome_prints_once(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(environment.sys.stderr, "isatty", lambda: True)

    welcome.maybe_show_welcome()
    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert captured.err.count("Thank you for using Giskard open-source!") == 1
