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

from posthog import Posthog, identify_context, tag

_DISABLING_ENV_VARS = [
    "DO_NOT_TRACK",
    "GISKARD_TELEMETRY_DISABLED",
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

    # Generate new persistent ID
    new_id = str(uuid.uuid4())
    try:
        _ = config_path.parent.mkdir(parents=True, exist_ok=True)
        _ = config_path.write_text(new_id, encoding="utf-8")
    except OSError:
        # Fallback for read-only systems
        return f"anon-{uuid.uuid4()}"
    return new_id


_anonymous_id = _get_or_create_anonymous_id()

telemetry = Posthog(
    project_api_key="phc_Asp36pe4X5WMqeJ4aMMV4gq5LGdGw69mdYSdEYGpbxm2",  # pragma: allowlist secret
    host="https://eu.i.posthog.com",
    disabled=_should_disable(),
)


def disable_telemetry() -> None:
    """
    Disable telemetry. Overrides the environment variable settings.
    """
    telemetry.disabled = True


# Nested ``telemetry_run_context`` / ``scoped_telemetry``: only the outermost
# scope emits ``giskard_uncaught_exception`` (one event per logical failure).
_telemetry_run_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_telemetry_run_depth", default=0
)


@contextmanager
def telemetry_run_context() -> Iterator[None]:
    """Open a PostHog context scope for a logical operation (sync or async body).

    Use inside ``async def`` with ``with telemetry_run_context():`` so nested
    ``scoped_telemetry`` calls get a consistent parent scope. Pair with
    ``telemetry_tag`` (see ``giskard.core``) to attach non-PII dimensions to
    child captures.
    """
    depth_token = _telemetry_run_depth.set(_telemetry_run_depth.get() + 1)
    try:
        with telemetry.new_context(capture_exceptions=False):
            if _anonymous_id is not None:
                identify_context(_anonymous_id)
            _set_tags()
            try:
                yield
            except Exception as e:
                # Do not send exception text: it may contain user content, secrets, or paths.
                if _telemetry_run_depth.get() == 1:
                    _ = telemetry.capture(
                        "giskard_uncaught_exception",
                        properties={
                            "exception_type": type(e).__name__,
                        },
                    )
                raise
    finally:
        _telemetry_run_depth.reset(depth_token)


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
