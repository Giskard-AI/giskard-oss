from giskard.core.settings import GiskardCoreSettings, get_settings


def test_hide_welcome_defaults_false():
    assert get_settings().hide_welcome is False


def test_hide_welcome_reads_env(monkeypatch):
    monkeypatch.setenv("GISKARD_HIDE_WELCOME", "true")
    assert GiskardCoreSettings().hide_welcome is True
