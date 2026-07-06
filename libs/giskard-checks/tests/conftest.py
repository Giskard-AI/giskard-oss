import importlib
import os

import giskard.checks.settings as settings_module
import pytest
from giskard.core import disable_telemetry

GISKARD_ENV_PREFIX = "GISKARD_"

_PROVIDER_PACKAGES = {
    "openai": "openai",
    "google": "google.genai",
    "anthropic": "anthropic",
    "litellm": "litellm",
}

_ANY_PROVIDER_PACKAGES = ["openai", "google.genai", "anthropic"]


def _is_installed(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def isolate_giskard_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """Isolate tests from ambient ``GISKARD_*`` configuration.

    Removes every ``GISKARD_``-prefixed environment variable and disables the
    ``.env`` file lookup so that tests observe the built-in defaults regardless
    of the developer's local environment (see issue #2734).

    Tests that genuinely depend on the ambient environment can opt out with the
    ``uses_ambient_env`` marker.
    """
    if request.node.get_closest_marker("uses_ambient_env"):
        yield
        return

    for name in [k for k in os.environ if k.startswith(GISKARD_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setitem(
        settings_module.GiskardChecksSettings.model_config, "env_file", None
    )
    yield


@pytest.fixture(autouse=True)
def reset_default_generator():
    """Restore the global default generator after each test."""
    original = settings_module._default_generator
    yield
    settings_module._default_generator = original


def pytest_configure(config: pytest.Config) -> None:
    """Disable telemetry for tests."""
    disable_telemetry()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI toggle for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless explicitly requested."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="Pass --run-integration to include integration tests."
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    installed_cache: dict[str, bool] = {}
    for item in items:
        for mark_name, package in _PROVIDER_PACKAGES.items():
            if mark_name in item.keywords:
                if package not in installed_cache:
                    installed_cache[package] = _is_installed(package)
                if not installed_cache[package]:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Provider SDK '{package}' not installed"
                        )
                    )

    if not any("no_providers" in item.keywords for item in items):
        return

    any_installed = any(_is_installed(p) for p in _ANY_PROVIDER_PACKAGES)
    if any_installed:
        for item in items:
            if "no_providers" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="no_providers tests require no provider SDKs installed"
                    )
                )


def pytest_sessionfinish(session, exitstatus):
    # If no tests were collected, set the exit status to 0 to avoid failure.
    # This is a workaround for packages not having any functional tests.
    if exitstatus == 5:
        session.exitstatus = 0
