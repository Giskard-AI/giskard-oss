from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from giskard.agents.embeddings.sentence_transformers_embedding import (
    SentenceTransformerEmbedding,
)


class TestSentenceTransformerEmbedding:
    """Tests for SentenceTransformerEmbedding."""

    def test_registered_discriminator(self):
        """Model should be registered with discriminator 'sentence-transformers'."""
        from giskard.agents.embeddings.base import BaseEmbeddingModel

        # Verify the model is registered
        assert "sentence-transformers" in BaseEmbeddingModel._discriminator_map()

    def test_default_model_name(self):
        """Default model should be all-MiniLM-L6-v2."""
        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.SentenceTransformerEmbedding._get_st_model"
        ):
            model = SentenceTransformerEmbedding()
            assert model.model == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        """Custom model name should be accepted."""
        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.SentenceTransformerEmbedding._get_st_model"
        ):
            model = SentenceTransformerEmbedding(model="all-mpnet-base-v2")
            assert model.model == "all-mpnet-base-v2"

    def test_import_error_without_sentence_transformers(self):
        """Should raise ImportError with helpful message when package missing."""
        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.importlib.import_module",
            side_effect=ImportError,
        ):
            with patch.object(
                SentenceTransformerEmbedding,
                "_get_st_model",
                side_effect=ImportError(
                    "The `sentence-transformers` package is required for "
                    "SentenceTransformerEmbedding. Install it with:\n"
                    "  pip install sentence-transformers\n"
                    "Or install giskard-agents with the local-embeddings extra:\n"
                    "  pip install giskard-agents[local-embeddings]"
                ),
            ):
                with pytest.raises(ImportError, match="sentence-transformers"):
                    SentenceTransformerEmbedding()

    @pytest.mark.sentence_transformers
    @pytest.mark.functional
    async def test_embed_produces_valid_vectors(self):
        """Integration test: embeddings should be valid numpy arrays."""
        try:
            model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        texts = ["Hello world", "This is a test"]
        embeddings = await model.embed(texts)

        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert isinstance(embeddings[1], np.ndarray)
        assert embeddings[0].shape[0] > 0
        assert embeddings[1].shape[0] > 0
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert embeddings[0].shape[0] == 384

    @pytest.mark.sentence_transformers
    @pytest.mark.functional
    async def test_embeddings_are_different_for_different_texts(self):
        """Embeddings for different texts should not be identical."""
        try:
            model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        emb_a, emb_b = await model.embed(["Paris is in France", "Hello world"])

        # Cosine similarity between these should not be 1.0
        similarity = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        )
        assert similarity < 0.99  # Different texts should have different embeddings

    async def test_embed_with_mock(self):
        """Test embedding with mocked sentence-transformers."""
        mock_st = MagicMock()
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        mock_instance.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.SentenceTransformerEmbedding._get_st_model",
            return_value=mock_st,
        ):
            model = SentenceTransformerEmbedding(model="test-model")
            texts = ["Hello, world!", "This is a test."]
            embeddings = await model._embed(texts)

            assert len(embeddings) == 2
            assert isinstance(embeddings[0], np.ndarray)
            assert isinstance(embeddings[1], np.ndarray)
            assert len(embeddings[0]) == 3
            assert np.isclose(embeddings[0], np.array([0.1, 0.2, 0.3])).all()
            assert np.isclose(embeddings[1], np.array([0.4, 0.5, 0.6])).all()

    async def test_batching_works_with_mock(self):
        """Test that embedding with multiple batches works."""
        mock_st = MagicMock()
        mock_instance = MagicMock()

        # Return 3 embeddings for 3 texts
        mock_instance.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        mock_st.return_value = mock_instance

        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.SentenceTransformerEmbedding._get_st_model",
            return_value=mock_st,
        ):
            model = SentenceTransformerEmbedding(model="test-model")
            texts = ["text1", "text2", "text3"]
            embeddings = await model._embed(texts)

            assert len(embeddings) == 3
            assert all(isinstance(e, np.ndarray) for e in embeddings)

    def test_serialization(self):
        """Test JSON serialization/deserialization."""
        from giskard.agents.embeddings.base import BaseEmbeddingModel

        with patch(
            "giskard.agents.embeddings.sentence_transformers_embedding.SentenceTransformerEmbedding._get_st_model"
        ):
            model = SentenceTransformerEmbedding(model="all-mpnet-base-v2")
            json_str = model.model_dump_json()
            deserialized = BaseEmbeddingModel.model_validate_json(json_str)

            assert isinstance(deserialized, SentenceTransformerEmbedding)
            assert deserialized.model == "all-mpnet-base-v2"
