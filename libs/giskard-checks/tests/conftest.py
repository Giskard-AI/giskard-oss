import importlib

import giskard.checks.settings as settings_module
import pytest
from giskard.core import disable_telemetry

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
