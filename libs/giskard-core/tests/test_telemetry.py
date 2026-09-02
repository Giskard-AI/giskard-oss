import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def test_exception_metadata_failure_emits_fixed_fallback(monkeypatch):
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


def test_exception_capture_failure_does_not_mask_original(monkeypatch):
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


def test_nested_context_captures_one_safe_exception_event(monkeypatch):
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
