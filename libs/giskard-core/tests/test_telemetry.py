import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

telemetry_mod = importlib.import_module("giskard.core.telemetry.telemetry")

_CORE_SRC = Path(__file__).resolve().parents[1] / "src"

_OPT_OUT_VARS = (
    "DO_NOT_TRACK",
    "GISKARD_TELEMETRY_DISABLED",
    "GISKARD_TELEMETRY_DISABLE_GEOIP",
)

_NETWORK_PROBE = r"""
import json

calls = []


def record(self, method, url, *args, **kwargs):
    calls.append({"method": method, "url": str(url)})
    raise RuntimeError("network blocked")


import requests

requests.Session.request = record

from giskard.core.telemetry.telemetry import (
    _anonymous_id,
    _should_disable,
    telemetry,
    telemetry_capture,
    telemetry_run_context,
)

with telemetry_run_context():
    telemetry_capture("repro_event")
    _ = telemetry.capture("direct_capture")
try:
    telemetry.flush(timeout_seconds=0.2)
except Exception:
    pass

print(
    json.dumps(
        {
            "should_disable": _should_disable(),
            "disabled": bool(telemetry.disabled),
            "send": bool(telemetry.send),
            "anonymous_id_is_none": _anonymous_id is None,
            "queue_size": telemetry.queue.qsize(),
            "consumer_alive": [c.is_alive() for c in (telemetry.consumers or [])],
            "http_urls": [c["url"] for c in calls],
        }
    )
)
"""


@pytest.fixture
def _enabled_home(tmp_path, monkeypatch):
    """Run the id logic against a temp home with telemetry not disabled."""
    monkeypatch.setattr(telemetry_mod, "_should_disable", lambda: False)
    monkeypatch.setattr(telemetry_mod.Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def _clean_opt_out_env(monkeypatch, tmp_path):
    """Ignore ambient process env and cwd ``.env`` so tests control the inputs."""
    for name in _OPT_OUT_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
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


@pytest.mark.parametrize(
    "value",
    ["1", "true", "YES", "on", " t ", '"1"', "'true'"],
)
def test_is_true_str_accepts_truthy_and_quoted_values(value):
    assert telemetry_mod._is_true_str(value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_is_true_str_rejects_non_truthy_values(value):
    assert telemetry_mod._is_true_str(value) is False


@pytest.mark.parametrize("var", ["GISKARD_TELEMETRY_DISABLED", "DO_NOT_TRACK"])
def test_should_disable_reads_process_env(var, _clean_opt_out_env, monkeypatch):
    monkeypatch.setenv(var, "1")
    assert telemetry_mod._should_disable() is True


def test_should_disable_false_when_unset(_clean_opt_out_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert telemetry_mod._should_disable() is False


def test_should_disable_reads_dotenv(_clean_opt_out_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GISKARD_TELEMETRY_DISABLED=1\n", encoding="utf-8")
    assert telemetry_mod._should_disable() is True


def test_should_disable_reads_quoted_dotenv_and_export(
    _clean_opt_out_env, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('export DO_NOT_TRACK="1"\n', encoding="utf-8")
    assert telemetry_mod._should_disable() is True


def test_should_disable_reads_dotenv_inline_comment(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_text(
        "GISKARD_TELEMETRY_DISABLED=1 # opt out\n", encoding="utf-8"
    )
    assert telemetry_mod._should_disable() is True


def test_should_disable_reads_quoted_dotenv_with_comment(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_text('DO_NOT_TRACK="1" # opt out\n', encoding="utf-8")
    assert telemetry_mod._should_disable() is True


def test_unquoted_hash_without_space_is_not_truthy(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_text(
        "GISKARD_TELEMETRY_DISABLED=1#comment\n", encoding="utf-8"
    )
    assert telemetry_mod._should_disable() is False


def test_non_utf8_dotenv_does_not_raise(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_bytes(b"\xff\xfeGISKARD_TELEMETRY_DISABLED=1\n")
    assert telemetry_mod._should_disable() is False


def test_latin1_dotenv_still_reads_ascii_opt_out(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_bytes(b"GISKARD_TELEMETRY_DISABLED=1\nNOTE=caf\xe9\n")
    assert telemetry_mod._should_disable() is True


def test_process_env_wins_over_dotenv(_clean_opt_out_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GISKARD_TELEMETRY_DISABLED=1\n", encoding="utf-8")
    monkeypatch.setenv("GISKARD_TELEMETRY_DISABLED", "false")
    assert telemetry_mod._should_disable() is False


def test_apply_env_opt_out_stops_sender(monkeypatch, _clean_opt_out_env):
    client = telemetry_mod.telemetry
    paused: list[bool] = []

    class _Consumer:
        def pause(self) -> None:
            paused.append(True)

    unregistered: list[object] = []
    monkeypatch.setattr(client, "disabled", False)
    monkeypatch.setattr(client, "send", True)
    monkeypatch.setattr(client, "disable_geoip", False)
    monkeypatch.setattr(client, "consumers", [_Consumer()])
    monkeypatch.setattr(
        telemetry_mod.atexit, "unregister", lambda fn: unregistered.append(fn)
    )
    monkeypatch.setenv("GISKARD_TELEMETRY_DISABLED", "1")

    telemetry_mod._apply_env_opt_out()

    assert client.disabled is True
    assert client.send is False
    assert client.disable_geoip is True
    assert paused == [True]
    assert unregistered == [client.join]
    assert client.consumers == []


def test_telemetry_capture_does_not_call_posthog_when_opted_out(
    monkeypatch, _clean_opt_out_env
):
    called: list[object] = []
    monkeypatch.setattr(
        telemetry_mod.telemetry,
        "capture",
        lambda *args, **kwargs: called.append(args) or "sent",
    )
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    token = telemetry_mod._in_telemetry_scope.set(True)
    try:
        telemetry_mod.telemetry_capture("should_not_send")
    finally:
        telemetry_mod._in_telemetry_scope.reset(token)

    assert called == []
    assert telemetry_mod.telemetry.disabled is True


def _probe_fresh_interpreter(
    tmp_path: Path, env_extra: dict[str, str], dotenv_text: str | None = None
) -> dict[str, object]:
    if dotenv_text is not None:
        (tmp_path / ".env").write_text(dotenv_text, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    for name in _OPT_OUT_VARS:
        env.pop(name, None)
    env.update(env_extra)
    env["HOME"] = str(home)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_CORE_SRC) if not existing else f"{_CORE_SRC}{os.pathsep}{existing}"
    )
    proc = subprocess.run(
        [sys.executable, "-c", _NETWORK_PROBE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    payload["id_file_exists"] = (home / ".giskard" / "id").exists()
    return payload


def test_opt_out_before_import_makes_no_http(tmp_path):
    """Firewalled hosts should see zero PostHog requests when opted out."""
    payload = _probe_fresh_interpreter(
        tmp_path, {"GISKARD_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1"}
    )
    assert payload["should_disable"] is True
    assert payload["disabled"] is True
    assert payload["send"] is False
    assert payload["anonymous_id_is_none"] is True
    assert payload["id_file_exists"] is False
    assert payload["http_urls"] == []
    assert payload["queue_size"] == 0
    consumer_alive = payload["consumer_alive"]
    assert isinstance(consumer_alive, list)
    assert all(item is False for item in consumer_alive)


def test_opt_out_via_dotenv_makes_no_http(tmp_path):
    payload = _probe_fresh_interpreter(
        tmp_path, {}, dotenv_text="GISKARD_TELEMETRY_DISABLED=1\n"
    )
    assert payload["should_disable"] is True
    assert payload["disabled"] is True
    assert payload["send"] is False
    assert payload["http_urls"] == []
    assert payload["queue_size"] == 0
    assert payload["id_file_exists"] is False


def test_opt_out_via_quoted_process_env_makes_no_http(tmp_path):
    payload = _probe_fresh_interpreter(tmp_path, {"GISKARD_TELEMETRY_DISABLED": '"1"'})
    assert payload["should_disable"] is True
    assert payload["disabled"] is True
    assert payload["send"] is False
    assert payload["http_urls"] == []
    assert payload["id_file_exists"] is False


def test_empty_do_not_track_does_not_disable(_clean_opt_out_env, monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "")
    assert telemetry_mod._should_disable() is False


def test_dotenv_last_assignment_wins(_clean_opt_out_env, tmp_path):
    (tmp_path / ".env").write_text(
        "GISKARD_TELEMETRY_DISABLED=0\nGISKARD_TELEMETRY_DISABLED=1\n",
        encoding="utf-8",
    )
    assert telemetry_mod._should_disable() is True


def test_empty_process_env_wins_over_dotenv(_clean_opt_out_env, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DO_NOT_TRACK=1\n", encoding="utf-8")
    monkeypatch.setenv("DO_NOT_TRACK", "")
    assert telemetry_mod._should_disable() is False


def test_apply_env_opt_out_is_one_way(monkeypatch, _clean_opt_out_env):
    client = telemetry_mod.telemetry
    monkeypatch.setattr(client, "disabled", False)
    monkeypatch.setattr(client, "send", True)
    monkeypatch.setattr(client, "disable_geoip", False)
    monkeypatch.setenv("GISKARD_TELEMETRY_DISABLED", "1")
    telemetry_mod._apply_env_opt_out()
    monkeypatch.delenv("GISKARD_TELEMETRY_DISABLED")

    telemetry_mod._apply_env_opt_out()

    assert client.disabled is True
    assert client.send is False


def test_geoip_only_opt_out(monkeypatch, _clean_opt_out_env):
    client = telemetry_mod.telemetry
    monkeypatch.setattr(client, "disabled", False)
    monkeypatch.setattr(client, "disable_geoip", False)
    monkeypatch.setenv("GISKARD_TELEMETRY_DISABLE_GEOIP", "1")

    telemetry_mod._apply_env_opt_out()

    assert client.disabled is False
    assert client.disable_geoip is True


_ATEXIT_PROBE = r"""
import json
import time

from giskard.core.telemetry.telemetry import disable_telemetry, telemetry

assert telemetry.send is True
assert telemetry.consumers
t0 = time.monotonic()
disable_telemetry()
telemetry.join()
elapsed = time.monotonic() - t0
print(
    json.dumps(
        {
            "send": bool(telemetry.send),
            "disabled": bool(telemetry.disabled),
            "consumers": list(telemetry.consumers or []),
            "elapsed": elapsed,
        }
    )
)
"""


def test_late_opt_out_join_returns_immediately(tmp_path):
    """Late disable must not let atexit ``join()`` wait on in-flight HTTP."""
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    for name in _OPT_OUT_VARS:
        env.pop(name, None)
    env["HOME"] = str(home)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_CORE_SRC) if not existing else f"{_CORE_SRC}{os.pathsep}{existing}"
    )
    proc = subprocess.run(
        [sys.executable, "-c", _ATEXIT_PROBE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["send"] is False
    assert payload["disabled"] is True
    assert payload["consumers"] == []
    assert payload["elapsed"] < 1.0
