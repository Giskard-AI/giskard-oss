import asyncio
import atexit
import contextvars
import functools
import hashlib
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import cast

from posthog import Posthog, identify_context, set_context_session, tag

from ..utils import GISKARD_LIBS_VERSIONS, is_true_env_str

_DISABLING_ENV_VARS = [
    "DO_NOT_TRACK",
    "GISKARD_TELEMETRY_DISABLED",
]
_DISABLE_GEOIP_ENV_VARS = [
    "GISKARD_TELEMETRY_DISABLE_GEOIP",
]
_EXCEPTION_TYPES: dict[type[Exception], str] = {
    AssertionError: "AssertionError",
    AttributeError: "AttributeError",
    ImportError: "ImportError",
    IndexError: "IndexError",
    KeyError: "KeyError",
    MemoryError: "MemoryError",
    NotImplementedError: "NotImplementedError",
    OSError: "OSError",
    RuntimeError: "RuntimeError",
    TimeoutError: "TimeoutError",
    TypeError: "TypeError",
    ValueError: "ValueError",
}
_GISKARD_COMPONENTS = frozenset({"agents", "checks", "core", "llm", "scan"})


def _get_component_roots() -> dict[str, tuple[Path, ...]]:
    core_root = Path(__file__).resolve().parents[1]
    namespace_root = core_root.parent
    roots: dict[str, tuple[Path, ...]] = {}
    for component in _GISKARD_COMPONENTS:
        installed_root = namespace_root / component
        candidates = [installed_root] if installed_root.is_dir() else []
        if (
            namespace_root.parent.name == "src"
            and namespace_root.parent.parent.name == "giskard-core"
        ):
            workspace_root = (
                namespace_root.parents[2]
                / f"giskard-{component}"
                / "src"
                / "giskard"
                / component
            )
            if workspace_root.is_dir():
                candidates.append(workspace_root)
        roots[component] = tuple(candidate.resolve() for candidate in candidates)
    return roots


_GISKARD_COMPONENT_ROOTS = _get_component_roots()
_UNKNOWN_EXCEPTION_PROPERTIES = {
    "exception_type": "other",
    "source_component": "unknown",
    "traceback_fingerprint": "unknown",
}
_OPT_OUT_ENV_KEYS = frozenset((*_DISABLING_ENV_VARS, *_DISABLE_GEOIP_ENV_VARS))


def _dotenv_flags() -> dict[str, str]:
    """Opt-out flags from a cwd ``.env``, the file giskard.checks settings already read.

    Minimal dotenv rules: ``export`` prefixes and unquoted `` # comment`` tails
    are dropped; ``errors="replace"`` so a non-UTF-8 file cannot abort import.
    """
    try:
        text = Path(".env").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    flags: dict[str, str] = {}
    for line in text.lstrip("\ufeff").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() in _OPT_OUT_ENV_KEYS:
            flags[key.strip()] = re.sub(r"\s+#.*", "", value).strip()
    return flags


def _flag_is_true(name: str) -> bool:
    """True if ``name`` is truthy in the process env, else in cwd ``.env``."""
    value = os.environ.get(name)
    if value is None:
        value = _dotenv_flags().get(name)
    return value is not None and is_true_env_str(value.strip().strip("'\""))


def _should_disable() -> bool:
    return any(_flag_is_true(var) for var in _DISABLING_ENV_VARS)


def _should_disable_geoip() -> bool:
    return _should_disable() or any(
        _flag_is_true(var) for var in _DISABLE_GEOIP_ENV_VARS
    )


ENV_INFORMATION: dict[str, str] = {}


def _get_environment_info() -> str:
    # Detect CI (standard across GH Actions, GitLab, Jenkins, etc.)
    is_ci = is_true_env_str(os.getenv("CI")) or is_true_env_str(os.getenv("TF_BUILD"))

    # Detect Colab
    is_colab = "google.colab" in sys.modules

    # Detect Kaggle
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

    if is_ci:
        return "ci"
    if is_colab:
        return "colab"
    if is_kaggle:
        return "kaggle"
    return "local"


def _get_env_information() -> dict[str, str]:
    if not ENV_INFORMATION:
        ENV_INFORMATION.update(
            {
                **{
                    f"{lib.replace('-', '_')}_version": lib_version
                    for lib, lib_version in GISKARD_LIBS_VERSIONS.items()
                },
                "environment": _get_environment_info(),
            }
        )
    return ENV_INFORMATION


def _set_tags() -> None:
    env_information = _get_env_information()
    for key, value in env_information.items():
        tag(key, value)


def _get_or_create_anonymous_id() -> str | None:
    if _should_disable():
        return None

    config_path = Path.home() / ".giskard" / "id"
    if config_path.exists():
        try:
            content = config_path.read_text(encoding="utf-8").strip()
            if content:
                return content
            # Delete the empty/truncated file so the creation block below can recreate it and persist a new ID.
            config_path.unlink(missing_ok=True)
        except OSError:
            # Unreadable path (permissions, race with deletion, etc.): mint ephemeral below.
            pass

    # Atomically create the file so concurrent first-run processes converge on one ID
    # rather than each persisting a different UUID.
    new_id = str(uuid.uuid4())
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(config_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # Lost the race; another process just wrote its ID. Read theirs.
        try:
            content = config_path.read_text(encoding="utf-8").strip()
            return content if content else f"anon-{uuid.uuid4()}"
        except OSError:
            return f"anon-{uuid.uuid4()}"
    except OSError:
        # Read-only system, etc.
        return f"anon-{uuid.uuid4()}"
    try:
        _ = os.write(fd, new_id.encode("utf-8"))
    finally:
        os.close(fd)
    return new_id


_anonymous_id = _get_or_create_anonymous_id()
# Distinguishes events from different invocations on the same machine in PostHog
# dashboards, while _anonymous_id keeps them all linked to the same user.
_process_session_id = str(uuid.uuid4())

# send=False when opted out: no consumer thread, no atexit join, no HTTP,
# so a firewalled ``eu.i.posthog.com`` is never contacted.
_disabled_at_import = _should_disable()
telemetry = Posthog(
    project_api_key="phc_Asp36pe4X5WMqeJ4aMMV4gq5LGdGw69mdYSdEYGpbxm2",  # pragma: allowlist secret
    host="https://eu.i.posthog.com",
    disabled=_disabled_at_import,
    disable_geoip=_should_disable_geoip(),
    send=not _disabled_at_import,
)


def disable_telemetry() -> None:
    """Disable telemetry for this process, overriding the environment variables.

    One-way. Stops the PostHog sender so no further requests reach the
    analytics host. Does not remove ``~/.giskard/id`` if it was already created.
    """
    telemetry.disabled = True
    telemetry.disable_geoip = True
    telemetry.send = False
    for consumer in telemetry.consumers or []:
        consumer.pause()
    # Consumers are daemon threads; with the atexit join unregistered they
    # cannot delay exit on uploads to a blocked host.
    atexit.unregister(telemetry.join)


def _apply_env_opt_out() -> None:
    """Honor opt-out flags set after import (notebooks, ``.env``); one-way."""
    if _should_disable():
        disable_telemetry()
    elif _should_disable_geoip():
        telemetry.disable_geoip = True


# Tracks whether we are currently inside any telemetry scope.
# Used so that nested telemetry_run_context / scoped_telemetry emit
# giskard_uncaught_exception only once per logical failure, and so
# telemetry_capture can drop events that would otherwise be "personless".
_in_telemetry_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_in_telemetry_scope", default=False
)


def telemetry_capture(
    event: str, *, properties: dict[str, object] | None = None
) -> None:
    """Capture a telemetry event, dropping it if no telemetry_run_context is active.

    Outside a context, PostHog assigns a random per-event UUID and marks the event
    "personless" — disconnecting it from the persistent anonymous ID and inflating
    user counts in the dashboard. Drop those events instead so future regressions
    don't pollute analytics.

    Parameters
    ----------
    event : str
        The event name to capture.
    properties : dict[str, object] or None
        Optional event properties, passed through to PostHog unchanged.
    """
    _apply_env_opt_out()
    if telemetry.disabled or not _in_telemetry_scope.get():
        return
    _ = telemetry.capture(event, properties=properties)


def _get_owned_frame_identifier(frame: FrameType) -> tuple[str, str] | None:
    module_name = frame.f_globals.get("__name__")
    if not isinstance(module_name, str):
        return None

    name_parts = module_name.split(".")
    if len(name_parts) < 2 or name_parts[0] != "giskard":
        return None
    component = name_parts[1]
    if component not in _GISKARD_COMPONENTS:
        return None

    module = sys.modules.get(module_name)
    if module is None or frame.f_globals is not vars(module):
        return None

    module_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(module_origin, str):
        return None

    frame_path = Path(frame.f_code.co_filename).resolve()
    module_path = Path(module_origin).resolve()
    if frame_path != module_path or not frame_path.is_file():
        return None

    for component_root in _GISKARD_COMPONENT_ROOTS[component]:
        try:
            relative_path = frame_path.relative_to(component_root)
        except ValueError:
            continue
        module_parts = relative_path.with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        expected_module_name = ".".join(("giskard", component, *module_parts))
        if module_name != expected_module_name:
            continue
        return component, relative_path.as_posix()
    return None


def _get_exception_properties(error: Exception) -> dict[str, object]:
    exception_type = _EXCEPTION_TYPES.get(type(error), "other")
    owned_frames: list[tuple[str, str]] = []
    traceback = error.__traceback__
    while traceback is not None:
        owned_frame = _get_owned_frame_identifier(traceback.tb_frame)
        if owned_frame is not None:
            owned_frames.append(owned_frame)
        traceback = traceback.tb_next

    if not owned_frames:
        return {
            **_UNKNOWN_EXCEPTION_PROPERTIES,
            "exception_type": exception_type,
        }

    fingerprint_input = "\n".join(
        f"{component}:{relative_path}" for component, relative_path in owned_frames
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]
    return {
        "exception_type": exception_type,
        "source_component": owned_frames[-1][0],
        "traceback_fingerprint": f"v1:{fingerprint}",
    }


def _capture_uncaught_exception(error: Exception) -> None:
    properties: dict[str, object]
    try:
        properties = _get_exception_properties(error)
    except Exception:
        properties = dict(_UNKNOWN_EXCEPTION_PROPERTIES)

    try:
        telemetry_capture("giskard_uncaught_exception", properties=properties)
    except Exception:
        # Telemetry is best-effort and must never replace the application error.
        pass


@contextmanager
def telemetry_run_context() -> Iterator[None]:
    """Open a PostHog context scope for a logical operation (sync or async body).

    Use as a with-statement inside an async def so nested scoped_telemetry calls
    share a consistent parent scope. Pair with telemetry_tag (from giskard.core)
    to attach non-PII dimensions to child captures.
    """
    is_outermost = not _in_telemetry_scope.get()
    token = _in_telemetry_scope.set(True)
    try:
        _apply_env_opt_out()
        with telemetry.new_context(capture_exceptions=False):
            if _anonymous_id is not None:
                identify_context(_anonymous_id)
            set_context_session(_process_session_id)
            _set_tags()
            try:
                yield
            except Exception as e:
                # Do not send exception text: it may contain user content, secrets, or paths.
                if is_outermost:
                    _capture_uncaught_exception(e)
                raise
    finally:
        _in_telemetry_scope.reset(token)


def scoped_telemetry[F: Callable[..., object]](func: F) -> F:
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            with telemetry_run_context():
                return cast(object, await func(*args, **kwargs))

        return cast(F, async_wrapper)

    @functools.wraps(func)
    def sync_wrapper(*args: object, **kwargs: object) -> object:
        with telemetry_run_context():
            return func(*args, **kwargs)

    return cast(F, sync_wrapper)
