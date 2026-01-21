import os

import pytest
from giskard.agents.embeddings import EmbeddingModel
from giskard.agents.embeddings.base import EmbeddingParams
from giskard.agents.generators import Generator


@pytest.fixture
async def generator():
    """Fixture providing a configured generator for tests."""
    return Generator(model=os.getenv("TEST_MODEL", "gemini/gemini-2.0-flash"))


@pytest.fixture
def embedding_model():
    """Fixture providing a configured embedding model for tests."""
    return EmbeddingModel(
        model=os.getenv("TEST_EMBEDDING_MODEL", "text-embedding-3-small"),
        params=EmbeddingParams(dimensions=1536),
    )
