from unittest.mock import patch

import numpy as np
import pytest
from giskard.agents.embeddings.base import BaseEmbeddingModel, EmbeddingParams
from giskard.agents.embeddings.litellm_embedding_model import LitellmEmbeddingModel


class MockEmbeddingResponse:
    """Mock response from litellm aembedding."""

    data: list[dict[str, list[float]]]

    def __init__(self, embeddings: list[list[float]]):
        self.data = [{"embedding": emb} for emb in embeddings]


@pytest.fixture
def mock_embedding_response():
    """Mock response from litellm aembedding."""
    # Return a mock response with the expected structure
    return MockEmbeddingResponse(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )


async def test_litellm_embedding_model_embed_with_mock(
    mock_embedding_response: MockEmbeddingResponse,
) -> None:
    """Test embedding with a mock response."""
    model = LitellmEmbeddingModel(model="test-model")

    with patch(
        "giskard.agents.embeddings.litellm_embedding_model.aembedding",
        return_value=mock_embedding_response,
    ):
        texts = ["Hello, world!", "This is a test."]
        embeddings = await model.embed(texts)

        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert isinstance(embeddings[1], np.ndarray)
        assert len(embeddings[0]) == 3
        assert len(embeddings[1]) == 3
        assert np.isclose(embeddings[0], np.array([0.1, 0.2, 0.3])).all()
        assert np.isclose(embeddings[1], np.array([0.4, 0.5, 0.6])).all()


@pytest.mark.functional
async def test_embedding_model_real_embedding(
    embedding_model: LitellmEmbeddingModel,
) -> None:
    """Test real embedding generation (requires API key)."""
    texts = ["Hello, world!", "This is a test."]
    embeddings = await embedding_model.embed(texts)

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert isinstance(embeddings[1], np.ndarray)
    # Check that embeddings have the expected dimensions
    assert len(embeddings[0]) > 0
    assert len(embeddings[1]) > 0


def test_batched_embeddings_simple() -> None:
    """Test basic batching behavior."""
    model = LitellmEmbeddingModel()
    texts = ["text1", "text2", "text3", "text4"]

    batches = list(
        model.batched_embeddings(texts, max_batch_size=2, max_total_chars=100)
    )

    # Should create 2 batches of 2 texts each
    assert len(batches) == 2
    assert batches[0] == ["text1", "text2"]
    assert batches[1] == ["text3", "text4"]


def test_batched_embeddings_with_char_limit() -> None:
    """Test batching with character limit."""
    model = LitellmEmbeddingModel()
    texts = ["short", "a bit longer text", "tiny"]

    batches = list(
        model.batched_embeddings(texts, max_batch_size=10, max_total_chars=20)
    )

    # First batch: "short" (5 chars)
    # Adding "a bit longer text" (17 chars) would exceed 20 total chars
    # So second batch starts with "a bit longer text"
    # Third batch: "tiny" (4 chars)
    assert len(batches) == 3
    assert batches[0] == ["short"]
    assert batches[1] == ["a bit longer text"]
    assert batches[2] == ["tiny"]


def test_batched_embeddings_truncate_long_text() -> None:
    """Test that overly long texts are truncated."""
    model = LitellmEmbeddingModel()
    long_text = "a" * 50  # 50 characters, well over the limit
    texts = [long_text, "short"]

    batches = list(
        model.batched_embeddings(texts, max_batch_size=2, max_total_chars=10)
    )

    # Long text should be truncated to max_total_chars
    assert len(batches) == 2
    assert len(batches[0][0]) == 10  # Truncated to 10 chars
    assert batches[1] == ["short"]


def test_batched_embeddings_custom_limits() -> None:
    """Test batching with custom limits passed to method."""
    model = LitellmEmbeddingModel()
    texts = ["text1", "text2", "text3"]

    # Override with smaller limits
    batches = list(
        model.batched_embeddings(texts, max_batch_size=2, max_total_chars=50)
    )

    assert len(batches) == 2
    assert batches[0] == ["text1", "text2"]
    assert batches[1] == ["text3"]


async def test_embed_with_multiple_batches() -> None:
    """Test that embed() correctly handles multiple batches."""
    model = LitellmEmbeddingModel(model="test-model")

    # Create a mock that returns different embeddings for each call
    call_count = 0

    async def mock_aembedding_side_effect(
        **kwargs: dict[str, object],
    ) -> MockEmbeddingResponse:
        nonlocal call_count
        call_count += 1
        # Return embeddings based on the batch
        input_list = kwargs.get("input", [])
        batch_size = len(input_list) if isinstance(input_list, list) else 0
        return MockEmbeddingResponse(
            embeddings=[
                [float(i + call_count * 0.1) for i in range(3)]
                for _ in range(batch_size)
            ]
        )

    with patch(
        "giskard.agents.embeddings.litellm_embedding_model.aembedding",
        side_effect=mock_aembedding_side_effect,
    ) as mock_aembedding:
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embeddings = await model.embed(texts, max_batch_size=2, max_total_chars=100)

        # With max_batch_size=2, we should have 3 batches: [2, 2, 1]
        assert mock_aembedding.call_count == 3
        assert len(embeddings) == 5
        assert all(isinstance(e, np.ndarray) for e in embeddings)


def test_embedding_model_serialization() -> None:
    """Test that embedding model can be serialized and deserialized."""
    model = LitellmEmbeddingModel(
        model="test-model",
        params=EmbeddingParams(dimensions=768),
    )

    # Serialize
    json_str = model.model_dump_json()

    # Deserialize
    deserialized_model = BaseEmbeddingModel.model_validate_json(json_str)

    assert isinstance(deserialized_model, LitellmEmbeddingModel)
    assert deserialized_model.model == "test-model"
    assert deserialized_model.params.dimensions == 768


def test_embedding_params_defaults() -> None:
    """Test that EmbeddingParams has sensible defaults."""
    params = EmbeddingParams()

    assert params.dimensions == 1536


def test_embedding_params_custom_values() -> None:
    """Test that EmbeddingParams accepts custom values."""
    params = EmbeddingParams(
        dimensions=512,
    )

    assert params.dimensions == 512


async def test_litellm_embedding_model_passes_params() -> None:
    """Test that custom params are passed to litellm aembedding."""
    model = LitellmEmbeddingModel(
        model="test-model",
        params=EmbeddingParams(dimensions=768),
    )

    mock_response = MockEmbeddingResponse(embeddings=[[0.1, 0.2, 0.3]])

    with patch(
        "giskard.agents.embeddings.litellm_embedding_model.aembedding",
        return_value=mock_response,
    ) as mock_aembedding:
        texts = ["test"]
        # Pass custom params via the params argument
        custom_params = EmbeddingParams(dimensions=512)
        _ = await model.embed(texts, params=custom_params)

        # Verify that aembedding was called with correct parameters
        mock_aembedding.assert_called_once()
        call_kwargs = mock_aembedding.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["input"] == ["test"]
        assert call_kwargs["dimensions"] == 512


def test_batched_embeddings_empty_list() -> None:
    """Test batching with empty list."""
    model = LitellmEmbeddingModel()
    texts: list[str] = []

    batches = list(model.batched_embeddings(texts))

    assert len(batches) == 0


def test_batched_embeddings_single_text() -> None:
    """Test batching with single text."""
    model = LitellmEmbeddingModel()
    texts = ["single text"]

    batches = list(model.batched_embeddings(texts))

    assert len(batches) == 1
    assert batches[0] == ["single text"]
