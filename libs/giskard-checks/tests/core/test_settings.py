import giskard.checks.settings as settings_module
import pytest
from giskard.agents import EmbeddingModel, Generator
from giskard.checks.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    get_default_embedding_model,
    get_default_generator,
    get_settings,
    set_default_generator,
)


def test_default_generator_uses_settings_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == "google/gemini-3.5-flash"


def test_default_generator_falls_back_to_builtin_default():
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, Generator)
    assert generator.model == DEFAULT_MODEL


def test_set_default_generator_overrides_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    explicit = Generator(model="anthropic/claude-haiku-4-5-20251001")

    set_default_generator(explicit)

    assert get_default_generator() is explicit


def test_default_embedding_model_uses_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == "google/gemini-embedding-001"


def test_default_embedding_model_falls_back_to_builtin_default():
    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, EmbeddingModel)
    assert embedding_model.model == DEFAULT_EMBEDDING_MODEL


def test_settings_max_reported_failures_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "3")
    assert get_settings().max_reported_failures == 3

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "invalid")
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "-1")
    assert get_settings().max_reported_failures is None

    monkeypatch.setenv("GISKARD_CHECKS_MAX_REPORTED_FAILURES", "true")
    assert get_settings().max_reported_failures is None


def test_settings_disable_rich_pretty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DISABLE_RICH_PRETTY", "true")
    assert get_settings().disable_rich_pretty is True
