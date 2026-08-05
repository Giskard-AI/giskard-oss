import importlib

import pytest

telemetry_mod = importlib.import_module("giskard.core.telemetry.telemetry")


@pytest.fixture
def _enabled_home(tmp_path, monkeypatch):
    """Run the id logic against a temp home with telemetry not disabled."""
    monkeypatch.setattr(telemetry_mod, "_should_disable", lambda: False)
    monkeypatch.setattr(telemetry_mod.Path, "home", lambda: tmp_path)
    return tmp_path


def test_anonymous_id_falls_back_on_empty_id_file(_enabled_home):
    """An empty/truncated ``~/.giskard/id`` (e.g. a crash between the atomic
    create and the write) must not collapse the anonymous id to ``""`` — the
    fast path should fall back to an ephemeral id, mirroring the race-loser
    ``FileExistsError`` branch."""
    id_path = _enabled_home / ".giskard" / "id"
    id_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.write_text("", encoding="utf-8")

    result = telemetry_mod._get_or_create_anonymous_id()

    assert result, "empty id file must not yield an empty anonymous id"


def test_anonymous_id_reads_existing_id_file(_enabled_home):
    """A populated id file is returned verbatim (stripped)."""
    id_path = _enabled_home / ".giskard" / "id"
    id_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.write_text("  existing-id\n", encoding="utf-8")

    assert telemetry_mod._get_or_create_anonymous_id() == "existing-id"
