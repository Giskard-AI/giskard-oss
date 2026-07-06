from unittest.mock import MagicMock, patch

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


@patch("giskard.agents.resolve.supports_native", return_value=True)
def test_get_default_generator_uses_native_when_supported(mock_supports):
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, GiskardLLMGenerator)
    assert generator.model == DEFAULT_MODEL
    mock_supports.assert_called_once_with(DEFAULT_MODEL, "completion")


@patch("giskard.agents.resolve.supports_native", return_value=True)
def test_get_default_embedding_model_uses_native_when_supported(mock_supports):
    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, GiskardLLMEmbeddingModel)
    assert embedding_model.model == DEFAULT_EMBEDDING_MODEL
    mock_supports.assert_called_once_with(DEFAULT_EMBEDDING_MODEL, "embedding")


@patch("giskard.agents.resolve.supports_native", return_value=True)
def test_get_default_generator_honors_settings_model_when_native(
    mock_supports, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    settings_module._default_generator = None

    generator = get_default_generator()

    assert isinstance(generator, GiskardLLMGenerator)
    assert generator.model == "google/gemini-3.5-flash"
    mock_supports.assert_called_once_with("google/gemini-3.5-flash", "completion")


@patch("giskard.agents.resolve.supports_native", return_value=True)
def test_get_default_embedding_model_honors_settings_when_native(
    mock_supports, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "google/gemini-embedding-001"
    )

    embedding_model = get_default_embedding_model()

    assert isinstance(embedding_model, GiskardLLMEmbeddingModel)
    assert embedding_model.model == "google/gemini-embedding-001"
    mock_supports.assert_called_once_with("google/gemini-embedding-001", "embedding")


@patch("giskard.agents.generators.litellm_generator.LiteLLMGenerator")
@patch("giskard.agents.resolve.supports_native", return_value=False)
def test_get_default_generator_falls_back_to_litellm_when_unsupported(
    mock_supports, mock_generator_cls
):
    expected = MagicMock(spec=LiteLLMGenerator)
    mock_generator_cls.return_value = expected
    settings_module._default_generator = None

    generator = get_default_generator()

    assert generator is expected
    mock_supports.assert_called_once_with(DEFAULT_MODEL, "completion")
    mock_generator_cls.assert_called_once_with(model=DEFAULT_MODEL)


@patch("giskard.agents.resolve.LiteLLMEmbeddingModel")
@patch("giskard.agents.resolve.supports_native", return_value=False)
def test_get_default_embedding_model_falls_back_to_litellm_when_unsupported(
    mock_supports, mock_embedding_cls
):
    expected = MagicMock(spec=LiteLLMEmbeddingModel)
    mock_embedding_cls.return_value = expected

    embedding_model = get_default_embedding_model()

    assert embedding_model is expected
    mock_supports.assert_called_once_with(DEFAULT_EMBEDDING_MODEL, "embedding")
    mock_embedding_cls.assert_called_once_with(model=DEFAULT_EMBEDDING_MODEL)


@patch("giskard.agents.generators.litellm_generator.LiteLLMGenerator")
@patch("giskard.agents.resolve.supports_native", return_value=False)
def test_get_default_generator_settings_model_falls_back_to_litellm(
    mock_supports, mock_generator_cls, monkeypatch: pytest.MonkeyPatch
):
    expected = MagicMock(spec=LiteLLMGenerator)
    mock_generator_cls.return_value = expected
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "deepseek/deepseek-chat")
    settings_module._default_generator = None

    generator = get_default_generator()

    assert generator is expected
    mock_supports.assert_called_once_with("deepseek/deepseek-chat", "completion")
    mock_generator_cls.assert_called_once_with(model="deepseek/deepseek-chat")


@patch("giskard.agents.resolve.LiteLLMEmbeddingModel")
@patch("giskard.agents.resolve.supports_native", return_value=False)
def test_get_default_embedding_model_settings_falls_back_to_litellm(
    mock_supports, mock_embedding_cls, monkeypatch: pytest.MonkeyPatch
):
    expected = MagicMock(spec=LiteLLMEmbeddingModel)
    mock_embedding_cls.return_value = expected
    monkeypatch.setenv(
        "GISKARD_CHECKS_DEFAULT_EMBEDDING_MODEL", "deepseek/deepseek-embed"
    )

    embedding_model = get_default_embedding_model()

    assert embedding_model is expected
    mock_supports.assert_called_once_with("deepseek/deepseek-embed", "embedding")
    mock_embedding_cls.assert_called_once_with(model="deepseek/deepseek-embed")


def test_set_default_generator_overrides_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GISKARD_CHECKS_DEFAULT_MODEL", "google/gemini-3.5-flash")
    explicit = Generator(model="anthropic/claude-haiku-4-5-20251001")

    set_default_generator(explicit)

    assert get_default_generator() is explicit


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
