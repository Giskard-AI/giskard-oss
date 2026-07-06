"""Tests for generator and embedding backend resolution."""

from unittest.mock import AsyncMock, MagicMock, patch

from giskard.agents.embeddings.litellm_embedding_model import LitellmEmbeddingModel
from giskard.agents.embeddings.litellm_package_embedding_model import (
    LiteLLMEmbeddingModel,
)
from giskard.agents.embeddings.resolve import resolve_embedding_model
from giskard.agents.generators.giskard_llm_generator import GiskardLLMGenerator
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
from giskard.agents.generators.resolve import resolve_generator


@patch("giskard.agents.generators.resolve.supports_native", return_value=True)
def test_resolve_generator_uses_giskard_llm_when_native_supported(mock_supports):
    generator = resolve_generator("openai/gpt-4o-mini")

    assert isinstance(generator, GiskardLLMGenerator)
    assert generator.model == "openai/gpt-4o-mini"
    mock_supports.assert_called_once_with("openai/gpt-4o-mini", "completion")


@patch("giskard.agents.generators.litellm_generator.LiteLLMGenerator")
@patch("giskard.agents.generators.resolve.supports_native", return_value=False)
def test_resolve_generator_uses_litellm_when_native_unsupported(
    mock_supports, mock_generator_cls
):
    expected = MagicMock(spec=LiteLLMGenerator)
    mock_generator_cls.return_value = expected

    generator = resolve_generator("deepseek/deepseek-chat")

    assert generator is expected
    mock_supports.assert_called_once_with("deepseek/deepseek-chat", "completion")
    mock_generator_cls.assert_called_once_with(model="deepseek/deepseek-chat")


@patch("giskard.agents.embeddings.resolve.supports_native", return_value=True)
def test_resolve_embedding_model_uses_giskard_llm_when_native_supported(
    mock_supports,
):
    model = resolve_embedding_model("text-embedding-3-small")

    assert isinstance(model, LitellmEmbeddingModel)
    assert model.model == "text-embedding-3-small"
    mock_supports.assert_called_once_with("text-embedding-3-small", "embedding")


@patch(
    "giskard.agents.embeddings.litellm_package_embedding_model.LiteLLMEmbeddingModel"
)
@patch("giskard.agents.embeddings.resolve.supports_native", return_value=False)
def test_resolve_embedding_model_uses_litellm_package_when_native_unsupported(
    mock_supports, mock_embedding_cls
):
    expected = MagicMock(spec=LiteLLMEmbeddingModel)
    mock_embedding_cls.return_value = expected

    model = resolve_embedding_model("deepseek/deepseek-embed")

    assert model is expected
    mock_supports.assert_called_once_with("deepseek/deepseek-embed", "embedding")
    mock_embedding_cls.assert_called_once_with(model="deepseek/deepseek-embed")


async def test_litellm_package_embedding_model_embed_with_mock(monkeypatch):
    model = LiteLLMEmbeddingModel(model="test-embedding")

    embedding = MagicMock()
    embedding.embedding = [0.1, 0.2, 0.3]
    response = MagicMock()
    response.data = [embedding]

    mock_litellm = MagicMock()
    mock_litellm.aembedding = AsyncMock(return_value=response)
    monkeypatch.setattr(
        "giskard.agents.embeddings.litellm_package_embedding_model._import_litellm",
        lambda: mock_litellm,
    )

    vectors = await model.embed(["hello"])

    assert len(vectors) == 1
    assert vectors[0].tolist() == [0.1, 0.2, 0.3]
    mock_litellm.aembedding.assert_awaited_once_with(
        model="test-embedding",
        input=["hello"],
        dimensions=1536,
    )
