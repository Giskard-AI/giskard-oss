import asyncio
import contextvars
import functools
import os
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from posthog import Posthog, identify_context, set_context_session, tag

_DISABLING_ENV_VARS = [
    "DO_NOT_TRACK",
    "GISKARD_TELEMETRY_DISABLED",
]
_DISABLE_GEOIP_ENV_VARS = [
    "GISKARD_TELEMETRY_DISABLE_GEOIP",
]
# Common truthy values used in CLI tools and web frameworks
_TRUTHY_VALUES = {"1", "true", "yes", "on", "t", "y"}


def _is_true_str(value: str | None) -> bool:
    if value is None:
        return False

    value = value.strip().lower()

    return value in _TRUTHY_VALUES


def _should_disable() -> bool:
    return any(_is_true_str(os.getenv(var)) for var in _DISABLING_ENV_VARS)


def _should_disable_geoip() -> bool:
    return _should_disable() or any(
        _is_true_str(os.getenv(var)) for var in _DISABLE_GEOIP_ENV_VARS
    )


def _get_lib_version(lib: str) -> str:
    try:
        return version(lib)
    except PackageNotFoundError:
        return "not_installed"


def _get_environment_info() -> str:
    # Detect CI (standard across GH Actions, GitLab, Jenkins, etc.)
    is_ci = _is_true_str(os.getenv("CI")) or _is_true_str(os.getenv("TF_BUILD"))

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


ENV_INFORMATION: dict[str, str] = {}


def _get_env_information() -> dict[str, str]:
    if not ENV_INFORMATION:
        ENV_INFORMATION.update(
            {
                "giskard_core_version": _get_lib_version("giskard-core"),
                "giskard_checks_version": _get_lib_version("giskard-checks"),
                "giskard_agents_version": _get_lib_version("giskard-agents"),
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
            return config_path.read_text(encoding="utf-8").strip()
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

telemetry = Posthog(
    project_api_key="phc_Asp36pe4X5WMqeJ4aMMV4gq5LGdGw69mdYSdEYGpbxm2",  # pragma: allowlist secret
    host="https://eu.i.posthog.com",
    disabled=_should_disable(),
    disable_geoip=_should_disable_geoip(),
)


def disable_telemetry() -> None:
    """
    Disable telemetry. Overrides the environment variable settings.
    """
    telemetry.disabled = True
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
    if not _in_telemetry_scope.get():
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
