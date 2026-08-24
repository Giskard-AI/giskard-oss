import pytest
from giskard.core import welcome


@pytest.fixture(autouse=True)
def reset_welcome_state():
    welcome._shown = False
    yield
    welcome._shown = False


@pytest.mark.parametrize(
    ("hide_welcome", "expected"),
    [
        (None, True),
        ("1", False),
        ("", True),
        ("maybe", True),
    ],
)
def test_should_show_welcome(monkeypatch, hide_welcome, expected):
    monkeypatch.delenv("GISKARD_HIDE_WELCOME", raising=False)
    if hide_welcome is not None:
        monkeypatch.setenv("GISKARD_HIDE_WELCOME", hide_welcome)

    assert welcome._should_show_welcome() is expected


def test_maybe_show_welcome_prints_to_stderr(capsys):
    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert "Thank you for using Giskard open-source!" in captured.err
    assert captured.out == ""
    assert welcome._shown is True


def test_maybe_show_welcome_prints_once(capsys):
    welcome.maybe_show_welcome()
    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert captured.err.count("Thank you for using Giskard open-source!") == 1


def test_maybe_show_welcome_does_not_raise_on_invalid_hide_welcome(monkeypatch, capsys):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", "maybe")

    welcome.maybe_show_welcome()
    captured = capsys.readouterr()

    assert "Thank you for using Giskard open-source!" in captured.err
    assert welcome._shown is True
