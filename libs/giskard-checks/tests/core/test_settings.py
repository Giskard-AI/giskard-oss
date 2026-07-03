import giskard.checks.settings as settings_module
import pytest
from giskard.agents import EmbeddingModel, Generator
from giskard.checks.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    clear_settings_cache,
    get_default_embedding_model,
    get_default_generator,
    get_settings,
    set_default_generator,
)


@pytest.fixture(autouse=True)
def reset_checks_settings():
    """Restore runtime overrides and settings cache after each test."""
    original_generator = settings_module._default_generator
    original_embedding = settings_module._default_embedding_model
    clear_settings_cache()
    yield
    settings_module._default_generator = original_generator
    settings_module._default_embedding_model = original_embedding
    clear_settings_cache()


def test_default_generator_uses_settings_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    clear_settings_cache()

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == "google/gemini-3.5-flash"


def test_default_generator_falls_back_to_builtin_default():
    settings_module._default_generator = None
    clear_settings_cache()

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == DEFAULT_MODEL


def test_set_default_generator_overrides_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    clear_settings_cache()
    explicit = Generator(model="anthropic/claude-haiku-4-5-20251001")

    set_default_generator(explicit)

    assert get_default_generator() is explicit


def test_default_embedding_model_uses_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )
    clear_settings_cache()

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == "google/gemini-embedding-001"


def test_default_embedding_model_falls_back_to_builtin_default():
    settings_module._default_embedding_model = None
    clear_settings_cache()

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == DEFAULT_EMBEDDING_MODEL


def test_settings_max_reported_failures_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "3")
    clear_settings_cache()
    assert get_settings().max_reported_failures == 3

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "invalid")
    clear_settings_cache()
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "-1")
    clear_settings_cache()
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "true")
    clear_settings_cache()
    assert get_settings().max_reported_failures is None


def test_settings_disable_rich_pretty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DISABLE_RICH_PRETTY", "true")
    clear_settings_cache()
    assert get_settings().disable_rich_pretty is True
