import importlib.util

import pytest
from giskard.core import disable_telemetry


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
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="Pass --run-integration to include integration tests."
    )
    skip_regorus = pytest.mark.skip(
        reason="regorus not installed (pip install 'giskard-checks[regorus]')",
    )
    regorus_available = importlib.util.find_spec("regorus") is not None
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
        if not regorus_available and "regorus" in item.keywords:
            item.add_marker(skip_regorus)


def pytest_sessionfinish(session, exitstatus):
    # If no tests were collected, set the exit status to 0 to avoid failure.
    # This is a workaround for packages not having any functional tests.
    if exitstatus == 5:
        session.exitstatus = 0
