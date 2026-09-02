import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
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
def _enabled_capture(monkeypatch):
    monkeypatch.setattr(telemetry_mod, "_apply_env_opt_out", lambda: None)
    monkeypatch.setattr(telemetry_mod.telemetry, "disabled", False)


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


def _captured_exception(call):
    try:
        call()
    except Exception as error:
        return error
    raise AssertionError("call did not raise")


def test_exception_properties_are_stable_for_same_owned_traceback(monkeypatch):
    monkeypatch.setattr(
        telemetry_mod,
        "tag",
        lambda *_: (_ for _ in ()).throw(ValueError("first secret")),
    )
    first = _captured_exception(telemetry_mod._set_tags)
    monkeypatch.setattr(
        telemetry_mod,
        "tag",
        lambda *_: (_ for _ in ()).throw(ValueError("second secret")),
    )
    second = _captured_exception(telemetry_mod._set_tags)

    first_properties = telemetry_mod._get_exception_properties(first)
    second_properties = telemetry_mod._get_exception_properties(second)

    assert first_properties == second_properties
    assert first_properties["exception_type"] == "ValueError"
    assert first_properties["source_component"] == "core"
    fingerprint = str(first_properties["traceback_fingerprint"])
    assert fingerprint.startswith("v1:")
    assert len(fingerprint) == 19
    assert int(fingerprint.removeprefix("v1:"), 16) >= 0


def test_exception_properties_distinguish_owned_traceback_sites(monkeypatch):
    monkeypatch.setattr(
        telemetry_mod, "tag", lambda *_: (_ for _ in ()).throw(ValueError("secret"))
    )
    from_set_tags = _captured_exception(telemetry_mod._set_tags)

    class InvalidEnvironmentValue:
        def strip(self):
            raise ValueError("secret")

    from_utils = _captured_exception(
        lambda: telemetry_mod.is_true_env_str(InvalidEnvironmentValue())
    )

    assert (
        telemetry_mod._get_exception_properties(from_set_tags)["traceback_fingerprint"]
        != telemetry_mod._get_exception_properties(from_utils)["traceback_fingerprint"]
    )


def test_exception_properties_ignore_all_external_values(monkeypatch):
    private_values = {
        "message": "PRIVATE_MESSAGE_SENTINEL",
        "filename": "/private/customer/PRIVATE_PATH_SENTINEL.py",
        "module": "PRIVATE_MODULE_SENTINEL",
        "function": "PRIVATE_FUNCTION_SENTINEL",
        "class": "PRIVATE_CLASS_SENTINEL",
        "local": "PRIVATE_LOCAL_SENTINEL",
        "source": "PRIVATE_SOURCE_SENTINEL",
    }

    def make_external(filename, module_name, function_name, message, local_value):
        namespace = {"__name__": module_name}
        source = (
            f"def {function_name}(*_):\n"
            f"    private_local = {local_value!r}\n"
            "    del private_local\n"
            f"    raise ValueError({message!r})"
        )
        exec(compile(source, filename, "exec"), namespace)
        return namespace[function_name]

    first_external = make_external(
        private_values["filename"],
        private_values["module"],
        private_values["function"],
        private_values["message"],
        private_values["local"],
    )
    second_external = make_external(
        "/different/private/path.py",
        "different.private.module",
        "different_private_function",
        "DIFFERENT_PRIVATE_MESSAGE",
        "DIFFERENT_PRIVATE_LOCAL",
    )
    monkeypatch.setattr(telemetry_mod, "tag", first_external)
    first = _captured_exception(telemetry_mod._set_tags)
    monkeypatch.setattr(telemetry_mod, "tag", second_external)
    second = _captured_exception(telemetry_mod._set_tags)

    first_properties = telemetry_mod._get_exception_properties(first)
    assert first_properties == telemetry_mod._get_exception_properties(second)
    serialized = repr(first_properties)
    assert all(value not in serialized for value in private_values.values())


def test_exception_type_allowlist_uses_exact_class_identity():
    class PrivateValueError(ValueError):
        pass

    error = _captured_exception(lambda: (_ for _ in ()).throw(PrivateValueError()))

    assert telemetry_mod._get_exception_properties(error)["exception_type"] == "other"


def test_exception_properties_reject_forged_owned_filename():
    assert telemetry_mod.__file__ is not None
    owned_path = Path(telemetry_mod.__file__).with_name("forged_private.py")
    source = "def raise_private():\n    raise RuntimeError('PRIVATE_FORGED_SENTINEL')"
    namespace = {"__name__": "giskard.core.forged_private"}
    exec(compile(source, str(owned_path), "exec"), namespace)

    error = _captured_exception(namespace["raise_private"])

    assert telemetry_mod._get_exception_properties(error) == {
        "exception_type": "RuntimeError",
        "source_component": "unknown",
        "traceback_fingerprint": "unknown",
    }


def test_exception_properties_reject_mismatched_module_origin(monkeypatch):
    assert telemetry_mod.__file__ is not None
    module_name = telemetry_mod.__name__
    fake_module = ModuleType(module_name)
    fake_module.__spec__ = importlib.util.spec_from_file_location(module_name, __file__)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    exec(
        compile(
            "def raise_private():\n    raise RuntimeError('PRIVATE_SENTINEL')",
            telemetry_mod.__file__,
            "exec",
        ),
        fake_module.__dict__,
    )

    error = _captured_exception(fake_module.raise_private)

    assert telemetry_mod._get_exception_properties(error) == {
        "exception_type": "RuntimeError",
        "source_component": "unknown",
        "traceback_fingerprint": "unknown",
    }


def test_exception_properties_reject_module_with_mismatched_globals(monkeypatch):
    fake_module = ModuleType("giskard.core.private")
    fake_module.__file__ = __file__
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    namespace = {"__name__": fake_module.__name__}
    error = None
    try:
        exec(
            compile("raise RuntimeError('PRIVATE_SENTINEL')", __file__, "exec"),
            namespace,
        )
    except RuntimeError as raised:
        error = raised

    assert error is not None
    assert (
        telemetry_mod._get_exception_properties(error)["traceback_fingerprint"]
        == "unknown"
    )


def test_exception_properties_reject_mutated_component_path(tmp_path, monkeypatch):
    private_package = tmp_path / "private"
    private_package.mkdir()
    private_module_path = private_package / "leaked.py"
    private_module_path.write_text(
        "def raise_private():\n    raise RuntimeError('PRIVATE_PATH_SENTINEL')\n",
        encoding="utf-8",
    )
    module_name = "giskard.core.leaked"
    spec = importlib.util.spec_from_file_location(module_name, private_module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    import giskard.core

    monkeypatch.setattr(
        giskard.core,
        "__path__",
        [*giskard.core.__path__, str(private_package)],
    )
    spec.loader.exec_module(module)

    error = _captured_exception(module.raise_private)

    assert telemetry_mod._get_exception_properties(error) == {
        "exception_type": "RuntimeError",
        "source_component": "unknown",
        "traceback_fingerprint": "unknown",
    }


def test_exception_properties_ignore_traceback_line_number(monkeypatch):
    monkeypatch.setattr(
        telemetry_mod, "tag", lambda *_: (_ for _ in ()).throw(ValueError("secret"))
    )
    error = _captured_exception(telemetry_mod._set_tags)
    original = telemetry_mod._get_exception_properties(error)
    traceback = error.__traceback__
    assert traceback is not None and traceback.tb_next is not None
    forged = type(traceback)(
        traceback.tb_next,
        traceback.tb_frame,
        traceback.tb_lasti,
        987_654_321,
    )

    assert (
        telemetry_mod._get_exception_properties(error.with_traceback(forged))
        == original
    )


def test_exception_metadata_failure_emits_fixed_fallback(monkeypatch, _enabled_capture):
    captured = []
    original = RuntimeError("PRIVATE_ORIGINAL_SENTINEL")
    monkeypatch.setattr(
        telemetry_mod,
        "_get_exception_properties",
        lambda _: (_ for _ in ()).throw(OSError("PRIVATE_METADATA_SENTINEL")),
    )
    monkeypatch.setattr(
        telemetry_mod.telemetry,
        "capture",
        lambda event, *, properties: captured.append((event, properties)),
    )

    with pytest.raises(RuntimeError) as caught:
        with telemetry_mod.telemetry_run_context():
            raise original

    assert caught.value is original
    assert captured == [
        (
            "giskard_uncaught_exception",
            {
                "exception_type": "other",
                "source_component": "unknown",
                "traceback_fingerprint": "unknown",
            },
        )
    ]
    assert "PRIVATE" not in repr(captured)


def test_exception_capture_failure_does_not_mask_original(
    monkeypatch, _enabled_capture
):
    original = RuntimeError("PRIVATE_ORIGINAL_SENTINEL")
    monkeypatch.setattr(
        telemetry_mod.telemetry,
        "capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("PRIVATE_CAPTURE_SENTINEL")
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        with telemetry_mod.telemetry_run_context():
            raise original

    assert caught.value is original


def test_nested_context_captures_one_safe_exception_event(
    monkeypatch, _enabled_capture
):
    captured = []
    monkeypatch.setattr(
        telemetry_mod.telemetry,
        "capture",
        lambda event, *, properties: captured.append((event, properties)),
    )
    private_message = "PRIVATE_NESTED_MESSAGE_SENTINEL"

    with pytest.raises(ValueError, match=private_message):
        with telemetry_mod.telemetry_run_context():
            with telemetry_mod.telemetry_run_context():
                raise ValueError(private_message)

    assert len(captured) == 1
    event, properties = captured[0]
    assert event == "giskard_uncaught_exception"
    assert properties["exception_type"] == "ValueError"
    assert properties["source_component"] == "core"
    assert str(properties["traceback_fingerprint"]).startswith("v1:")
    assert private_message not in repr(captured)


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
        b"GISKARD_"
        + b"TELEMETRY_DISABLED=0\n"
        + b"GISKARD_"
        + b"TELEMETRY_DISABLED=1\n",
        b"GISKARD_TELEMETRY_DISABLED=1\nNOTE=caf\xe9\n",  # pragma: allowlist secret
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
