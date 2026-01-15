import pytest


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
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
