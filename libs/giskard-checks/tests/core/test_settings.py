import giskard.checks.settings as settings_module
import pytest
from giskard.agents import Generator
from giskard.agents.embeddings.giskard_llm_embedding_model import (
    GiskardLLMEmbeddingModel,
)
from giskard.agents.embeddings.litellm_embedding_model import LiteLLMEmbeddingModel
from giskard.agents.generators.giskard_llm_generator import GiskardLLMGenerator
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
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

    assert isinstance(generator, GiskardLLMGenerator)
    assert generator.model == "google/gemini-3.5-flash"


def test_default_generator_falls_back_to_builtin_default():
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, GiskardLLMGenerator)
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

    assert isinstance(embedding_model, GiskardLLMEmbeddingModel)
    assert embedding_model.model == "google/gemini-embedding-001"


def test_default_embedding_model_falls_back_to_builtin_default():
    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, GiskardLLMEmbeddingModel)
    assert embedding_model.model == DEFAULT_EMBEDDING_MODEL


def test_default_generator_uses_litellm_for_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("litellm")
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "deepseek/deepseek-chat")
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "deepseek/deepseek-chat"


def test_default_embedding_model_uses_litellm_for_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("litellm")
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "deepseek/deepseek-embed"
    )

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, LiteLLMEmbeddingModel)
    assert embedding_model.model == "deepseek/deepseek-embed"


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
