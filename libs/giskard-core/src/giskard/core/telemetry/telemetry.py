import asyncio
import atexit
import contextvars
import functools
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
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
_OPT_OUT_ENV_KEYS = frozenset((*_DISABLING_ENV_VARS, *_DISABLE_GEOIP_ENV_VARS))
# python-dotenv: unquoted inline comments need whitespace before ``#``.
_UNQUOTED_INLINE_COMMENT = re.compile(r"\s+#.*")


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _dotenv_value(raw: str) -> str:
    """Return a dotenv assignment value (quotes and unquoted `` #`` comments)."""
    val = raw.strip()
    if len(val) >= 2 and val[0] in {'"', "'"}:
        quote = val[0]
        end = val.find(quote, 1)
        if end != -1:
            return val[1:end]
    return _UNQUOTED_INLINE_COMMENT.sub("", val).rstrip()


def _is_true_str(value: str | None) -> bool:
    if value is None:
        return False
    return is_true_env_str(_strip_wrapping_quotes(value.strip()))


def _parse_dotenv(path: Path) -> dict[str, str]:
    try:
        # ``errors="replace"`` so a Latin-1/UTF-16 cwd ``.env`` cannot abort
        # ``import giskard``. ASCII opt-out keys still parse.
        text = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    except OSError:
        return {}

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in _OPT_OUT_ENV_KEYS:
            values[key] = _dotenv_value(val)
    return values


def _lookup_env(name: str) -> str | None:
    """Return a telemetry flag from the process env, else cwd ``.env``.

    Process environment wins so an exported opt-out is not overridden by a
    leftover ``.env`` value. ``giskard.checks`` already reads cwd ``.env`` for
    ``GISKARD_CHECKS_*`` settings, so the same file is consulted here without
    mutating ``os.environ``.
    """
    if name in os.environ:
        return os.environ[name]
    path = Path(".env")
    if not path.is_file():
        return None
    return _parse_dotenv(path).get(name)


def _should_disable() -> bool:
    return any(_is_true_str(_lookup_env(var)) for var in _DISABLING_ENV_VARS)


def _should_disable_geoip() -> bool:
    return _should_disable() or any(
        _is_true_str(_lookup_env(var)) for var in _DISABLE_GEOIP_ENV_VARS
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

# ``send=False`` skips the consumer thread and atexit join so a firewalled
# process never opens ``eu.i.posthog.com`` (blocked hosts otherwise surface as
# upload errors / flush timeouts even when no events should be sent).
_disabled_at_import = _should_disable()
telemetry = Posthog(
    project_api_key="phc_Asp36pe4X5WMqeJ4aMMV4gq5LGdGw69mdYSdEYGpbxm2",  # pragma: allowlist secret
    host="https://eu.i.posthog.com",
    disabled=_disabled_at_import,
    disable_geoip=_should_disable_geoip(),
    send=not _disabled_at_import,
)


def disable_telemetry() -> None:
    """Disable telemetry for this process.

    Overrides environment variable settings. Stops the PostHog sender so no
    further requests are made to the analytics host. Pauses consumers and
    clears them so atexit ``join`` does not wait on in-flight uploads (for
    example when ``eu.i.posthog.com`` is blocked). Does not remove
    ``~/.giskard/id`` if it was already created.
    """
    telemetry.disabled = True
    telemetry.disable_geoip = True
    telemetry.send = False
    consumers = telemetry.consumers
    if consumers:
        for consumer in consumers:
            consumer.pause()
        # Drop the list so atexit ``Client.join`` does not ``Thread.join`` an
        # in-flight upload. ``atexit.unregister(telemetry.join)`` is best-effort:
        # CPython 3.13 compares the registered bound method by identity, so it
        # often does not match a newly created ``telemetry.join``.
        telemetry.consumers = []
    atexit.unregister(telemetry.join)


def _apply_env_opt_out() -> None:
    """Honor opt-out flags set after import (notebooks, ``.env``, ``os.environ``).

    Disable is one-way: unsetting the flags later does not re-enable sending.
    """
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
                    telemetry_capture(
                        "giskard_uncaught_exception",
                        properties={"exception_type": type(e).__name__},
                    )
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
