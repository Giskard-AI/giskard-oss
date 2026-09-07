import importlib
import os

import pytest
from giskard.agents.embeddings import EmbeddingModel
from giskard.agents.embeddings.base import EmbeddingParams
from giskard.agents.generators import Generator
from giskard.core import disable_telemetry


def pytest_configure(config: pytest.Config) -> None:
    """Disable telemetry for tests."""
    disable_telemetry()


_PROVIDER_PACKAGES = {
    "openai": "openai",
    "google": "google.genai",
    "anthropic": "anthropic",
    "litellm": "litellm",
}


def _is_installed(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-skip tests marked with a provider whose SDK is not installed."""
    cache: dict[str, bool] = {}
    for item in items:
        for mark_name, package in _PROVIDER_PACKAGES.items():
            if mark_name in item.keywords:
                if package not in cache:
                    cache[package] = _is_installed(package)
                if not cache[package]:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Provider SDK '{package}' not installed"
                        )
                    )


@pytest.fixture(autouse=True)
def _clear_llm_provider_cache():
    """Reset cached provider instances between tests.

    The default client caches providers whose async clients are bound to
    the current event loop.  pytest-asyncio creates a new loop per test,
    so stale providers would hit ``Event loop is closed``.
    """
    from giskard.llm import reset

    yield
    reset()


@pytest.fixture
async def generator():
    """Fixture providing a configured generator for tests."""
    return Generator(model=os.getenv("TEST_MODEL", "google/gemini-3.5-flash"))


@pytest.fixture
async def litellm_generator():
    """Fixture providing a LiteLLM-backed generator.

    Requires the ``litellm`` optional extra. Tests using this fixture should
    be marked with ``@pytest.mark.litellm`` so they auto-skip when litellm
    is not installed.
    """
    from giskard.agents.generators.litellm_generator import LiteLLMGenerator

    return LiteLLMGenerator(
        model=os.getenv("TEST_LITELLM_MODEL", "gemini/gemini-3.5-flash")
    )


@pytest.fixture
def embedding_model():
    """Fixture providing a configured embedding model for tests."""
    return EmbeddingModel(
        model=os.getenv("TEST_EMBEDDING_MODEL", "google/gemini-embedding-001"),
        params=EmbeddingParams(dimensions=1536),
    )
