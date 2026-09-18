import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

telemetry_mod = importlib.import_module("giskard.core.telemetry.telemetry")

_CORE_SRC = Path(__file__).resolve().parents[1] / "src"

_OPT_OUT_VARS = (
    "DO_NOT_TRACK",
    "GISKARD_TELEMETRY_DISABLED",
    "GISKARD_TELEMETRY_DISABLE_GEOIP",
)


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


@pytest.mark.parametrize("var", ["GISKARD_TELEMETRY_DISABLED", "DO_NOT_TRACK"])
@pytest.mark.parametrize("value", ["1", "true", '"1"'])
def test_should_disable_reads_process_env(var, value, _clean_opt_out_env, monkeypatch):
    monkeypatch.setenv(var, value)
    assert telemetry_mod._should_disable() is True


@pytest.mark.parametrize(
    "content",
    [
        b"GISKARD_TELEMETRY_DISABLED=1\n",
        b'export DO_NOT_TRACK="1"\n',
        b"GISKARD_TELEMETRY_DISABLED=1 # opt out\n",
        b'DO_NOT_TRACK="1" # opt out\n',
        b"GISKARD_TELEMETRY_DISABLED=0\nGISKARD_TELEMETRY_DISABLED=1\n",
        b"GISKARD_TELEMETRY_DISABLED=1\nNOTE=caf\xe9\n",  # latin-1 elsewhere
    ],
)
def test_should_disable_reads_dotenv(content, _clean_opt_out_env):
    (_clean_opt_out_env / ".env").write_bytes(content)
    assert telemetry_mod._should_disable() is True


@pytest.mark.parametrize(
    "content",
    [
        None,  # no .env file
        b"DO_NOT_TRACK=0\n",
        b"GISKARD_TELEMETRY_DISABLED=1#comment\n",  # no space: not a comment
        b"\xff\xfeGISKARD_TELEMETRY_DISABLED=1\n",  # utf-16: unreadable, no crash
    ],
)
def test_should_disable_false_cases(content, _clean_opt_out_env):
    if content is not None:
        (_clean_opt_out_env / ".env").write_bytes(content)
    assert telemetry_mod._should_disable() is False


@pytest.mark.parametrize("env_value", ["false", ""])
def test_process_env_wins_over_dotenv(env_value, _clean_opt_out_env, monkeypatch):
    (_clean_opt_out_env / ".env").write_text(
        "GISKARD_TELEMETRY_DISABLED=1\n", encoding="utf-8"
    )
    monkeypatch.setenv("GISKARD_TELEMETRY_DISABLED", env_value)
    assert telemetry_mod._should_disable() is False


@pytest.mark.parametrize("channel", ["process-env", "dotenv"])
def test_late_opt_out_stops_sender_and_is_one_way(
    channel, monkeypatch, _clean_opt_out_env
):
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
    if channel == "process-env":
        monkeypatch.setenv("GISKARD_TELEMETRY_DISABLED", "1")
    else:
        (_clean_opt_out_env / ".env").write_text(
            "GISKARD_TELEMETRY_DISABLED=1\n", encoding="utf-8"
        )

    telemetry_mod._apply_env_opt_out()

    assert client.disabled is True
    assert client.send is False
    assert client.disable_geoip is True
    assert paused == [True]
    assert unregistered == [client.join]

    # One-way: removing the flag does not re-enable sending.
    if channel == "process-env":
        monkeypatch.delenv("GISKARD_TELEMETRY_DISABLED")
    else:
        (_clean_opt_out_env / ".env").unlink()
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


def test_telemetry_capture_does_not_call_posthog_when_opted_out(
    monkeypatch, _clean_opt_out_env
):
    called: list[object] = []
    monkeypatch.setattr(
        telemetry_mod.telemetry,
        "capture",
        lambda *args, **kwargs: called.append(args) or "sent",
    )
    monkeypatch.setattr(telemetry_mod.telemetry, "consumers", [])
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    token = telemetry_mod._in_telemetry_scope.set(True)
    try:
        telemetry_mod.telemetry_capture("should_not_send")
    finally:
        telemetry_mod._in_telemetry_scope.reset(token)

    assert called == []
    assert telemetry_mod.telemetry.disabled is True


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


def _run_probe(probe: str, tmp_path: Path, env_extra: dict[str, str]) -> dict[str, Any]:
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
        [sys.executable, "-c", probe],
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


@pytest.mark.parametrize(
    ("env_extra", "dotenv_text"),
    [
        ({"GISKARD_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1"}, None),
        ({"GISKARD_TELEMETRY_DISABLED": '"1"'}, None),
        ({}, "GISKARD_TELEMETRY_DISABLED=1\n"),
    ],
    ids=["process-env", "quoted-process-env", "dotenv"],
)
def test_opt_out_before_import_makes_no_http(tmp_path, env_extra, dotenv_text):
    """Firewalled hosts should see zero PostHog requests when opted out."""
    if dotenv_text is not None:
        (tmp_path / ".env").write_text(dotenv_text, encoding="utf-8")
    payload = _run_probe(_NETWORK_PROBE, tmp_path, env_extra)
    assert payload["should_disable"] is True
    assert payload["disabled"] is True
    assert payload["send"] is False
    assert payload["anonymous_id_is_none"] is True
    assert payload["id_file_exists"] is False
    assert payload["http_urls"] == []
    assert payload["queue_size"] == 0
    assert all(alive is False for alive in payload["consumer_alive"])


_ENABLED_PROBE = r"""
import json
import time

calls = []


def record(self, method, url, *args, **kwargs):
    calls.append(str(url))
    raise RuntimeError("network blocked")


import requests

requests.Session.request = record

from giskard.core.telemetry.telemetry import (
    telemetry,
    telemetry_capture,
    telemetry_run_context,
)

for consumer in telemetry.consumers:
    consumer.flush_interval = 0.2  # shorten the 5s batching window

with telemetry_run_context():
    telemetry_capture("enabled_event")

deadline = time.monotonic() + 10
while not calls and time.monotonic() < deadline:
    time.sleep(0.05)
print(
    json.dumps(
        {
            "send": bool(telemetry.send),
            "disabled": bool(telemetry.disabled),
            "urls": calls,
        }
    )
)
"""


def test_enabled_telemetry_still_sends(tmp_path):
    """Guard the opposite direction: with no opt-out flag an upload to the
    PostHog host must be attempted."""
    payload = _run_probe(_ENABLED_PROBE, tmp_path, {})
    assert payload["send"] is True
    assert payload["disabled"] is False
    assert any("eu.i.posthog.com" in url for url in payload["urls"])
    assert payload["id_file_exists"] is True


_LATE_OPT_OUT_PROBE = r"""
import json
import time

import requests


def hang(self, *args, **kwargs):
    time.sleep(60)
    raise RuntimeError("unreachable")


requests.Session.request = hang

from giskard.core.telemetry.telemetry import (
    disable_telemetry,
    telemetry,
    telemetry_capture,
    telemetry_run_context,
)

assert telemetry.send is True
with telemetry_run_context():
    telemetry_capture("queued_before_opt_out")
disable_telemetry()
print(
    json.dumps(
        {
            "send": bool(telemetry.send),
            "disabled": bool(telemetry.disabled),
            "running": [c.running for c in telemetry.consumers],
        }
    )
)
"""


def test_late_opt_out_does_not_hang_exit(tmp_path):
    """An event queued to a blocked host before a late opt-out must not make
    process exit wait on the upload; the 20s subprocess timeout is the check."""
    payload = _run_probe(_LATE_OPT_OUT_PROBE, tmp_path, {})
    assert payload["send"] is False
    assert payload["disabled"] is True
    assert payload["running"] == [False]
