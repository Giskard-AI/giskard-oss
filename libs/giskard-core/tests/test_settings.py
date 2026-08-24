import pytest
from giskard.core.settings import GiskardCoreSettings, get_settings


def test_hide_welcome_defaults_false():
    assert get_settings().hide_welcome is False


def test_hide_welcome_reads_env(monkeypatch):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", "true")
    assert GiskardCoreSettings().hide_welcome is True


@pytest.mark.parametrize("value", ["on", "YES", "1", " TRUE "])
def test_hide_welcome_accepts_truthy_env(monkeypatch, value):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", value)
    assert GiskardCoreSettings().hide_welcome is True


@pytest.mark.parametrize("value", ["", " ", "maybe", "enabled", "false", "0"])
def test_hide_welcome_empty_or_invalid_is_false(monkeypatch, value):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", value)
    assert GiskardCoreSettings().hide_welcome is False
