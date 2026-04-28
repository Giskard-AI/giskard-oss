"""Functional test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _clear_llm_provider_cache():
    """Reset cached provider instances between tests.

    ``giskard.llm.routing._default_client`` caches providers whose async
    clients are bound to the current event loop.  pytest-asyncio creates a
    new loop per test, so stale providers would hit ``Event loop is closed``.
    """
    from giskard.llm.routing import reset

    yield
    reset()
