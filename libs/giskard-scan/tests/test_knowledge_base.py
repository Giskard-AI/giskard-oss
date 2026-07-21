import asyncio
from typing import override

import numpy as np
import pytest
from giskard.agents.embeddings.base import BaseEmbeddingModel, EmbeddingParams
from giskard.scan.utils.knowledge_base import (
    Document,
    KnowledgeBase,
    normalize_knowledge_base,
)
from pydantic import ValidationError


class _StubEmbeddingModel(BaseEmbeddingModel):
    embeddings: list[list[float]]
    calls: list[list[str]]
    delay: float = 0.0

    @override
    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append(texts)
        return [np.asarray(embedding) for embedding in self.embeddings[: len(texts)]]


def test_knowledge_base_from_texts_does_not_compute_embeddings():
    knowledge_base = KnowledgeBase.from_texts(["alpha", "beta"])

    assert [document.content for document in knowledge_base.documents] == [
        "alpha",
        "beta",
    ]
    assert [document.embeddings for document in knowledge_base.documents] == [
        None,
        None,
    ]


def test_normalize_knowledge_base_accepts_existing_instance_and_texts():
    knowledge_base = KnowledgeBase.from_texts(["alpha"])
    normalized = normalize_knowledge_base(["beta"])

    assert normalize_knowledge_base(knowledge_base) is knowledge_base
    assert normalize_knowledge_base(None) is None
    assert isinstance(normalized, KnowledgeBase)
    assert normalized.documents[0].content == "beta"


def test_normalize_knowledge_base_rejects_single_string():
    """A bare str must raise, not become one document per character."""
    with pytest.raises(TypeError, match="not a single string"):
        normalize_knowledge_base("my document")  # pyright: ignore[reportArgumentType]


def test_knowledge_base_drops_empty_documents():
    knowledge_base = KnowledgeBase.from_texts(["valid", "  ", "also valid"])

    assert [document.content for document in knowledge_base.documents] == [
        "valid",
        "also valid",
    ]


def test_knowledge_base_rejects_all_empty_documents():
    with pytest.raises(ValueError, match="at least one non-empty document"):
        _ = KnowledgeBase.from_texts(["", "  "])


async def test_closest_documents_computes_missing_embeddings_lazily_in_batch():
    embedding_model = _StubEmbeddingModel(
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        calls=[],
    )
    knowledge_base = KnowledgeBase.from_texts(["seed", "near", "far"]).model_copy(
        update={"embedding_model": embedding_model}
    )

    assert embedding_model.calls == []

    closest = await knowledge_base.closest_documents(seed_index=0, max_documents=2)

    assert embedding_model.calls == [["seed", "near", "far"]]
    assert [document.content for document in closest] == ["seed", "near"]
    assert [document.embeddings for document in closest] == [
        [1.0, 0.0],
        [0.9, 0.1],
    ]
    assert [document.embeddings for document in knowledge_base.documents] == [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]


async def test_closest_documents_initializes_embeddings_once_for_concurrent_callers():
    embedding_model = _StubEmbeddingModel(
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        calls=[],
        delay=0.01,
    )
    knowledge_base = KnowledgeBase.from_texts(["seed", "near", "far"]).model_copy(
        update={"embedding_model": embedding_model}
    )

    results = await asyncio.gather(
        knowledge_base.closest_documents(seed_index=0, max_documents=2),
        knowledge_base.closest_documents(seed_index=1, max_documents=2),
        knowledge_base.closest_documents(seed_index=2, max_documents=2),
    )

    assert embedding_model.calls == [["seed", "near", "far"]]
    assert [[document.content for document in result] for result in results] == [
        ["seed", "near"],
        ["near", "seed"],
        ["far", "near"],
    ]


async def test_closest_documents_recomputes_all_embeddings_when_any_is_missing():
    embedding_model = _StubEmbeddingModel(
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        calls=[],
    )
    knowledge_base = KnowledgeBase(
        documents=(
            Document(content="seed", embeddings=[0.0, 1.0]),
            Document(content="near"),
        ),
        embedding_model=embedding_model,
    )

    closest = await knowledge_base.closest_documents(seed_index=0, max_documents=2)

    assert embedding_model.calls == [["seed", "near"]]
    assert [document.content for document in closest] == ["seed", "near"]
    assert [document.embeddings for document in closest] == [
        [1.0, 0.0],
        [0.9, 0.1],
    ]
    assert [document.embeddings for document in knowledge_base.documents] == [
        [1.0, 0.0],
        [0.9, 0.1],
    ]


async def test_closest_documents_to_text_embeds_query_against_cached_documents():
    embedding_model = _StubEmbeddingModel(
        embeddings=[[0.0, 1.0]],
        calls=[],
    )
    knowledge_base = KnowledgeBase(
        documents=(
            Document(content="seed", embeddings=[1.0, 0.0]),
            Document(content="match", embeddings=[0.0, 1.0]),
        ),
        embedding_model=embedding_model,
    )

    closest = await knowledge_base.closest_documents_to_text(
        text="query",
        max_documents=1,
    )

    assert embedding_model.calls == [["query"]]
    assert [document.content for document in closest] == ["match"]


async def test_closest_documents_to_text_rejects_empty_query_without_embedding():
    embedding_model = _StubEmbeddingModel(
        embeddings=[[0.0, 1.0]],
        calls=[],
    )
    knowledge_base = KnowledgeBase(
        documents=(Document(content="seed", embeddings=[1.0, 0.0]),),
        embedding_model=embedding_model,
    )

    with pytest.raises(ValueError, match="Query text must not be empty"):
        await knowledge_base.closest_documents_to_text(text="  \n\t", max_documents=1)

    assert embedding_model.calls == []


async def test_closest_documents_rejects_non_finite_embeddings():
    """NaN/Inf embeddings must raise, not slip past the zero-vector guard."""
    knowledge_base = KnowledgeBase(
        documents=(
            Document(content="seed", embeddings=[float("nan"), 0.0]),
            Document(content="near", embeddings=[1.0, 0.0]),
        ),
    )

    with pytest.raises(ValueError, match="non-finite"):
        await knowledge_base.closest_documents(seed_index=0, max_documents=2)


def test_knowledge_base_documents_are_immutable():
    knowledge_base = KnowledgeBase.from_texts(["alpha"])

    assert isinstance(knowledge_base.documents, tuple)
    with pytest.raises(ValidationError):
        setattr(
            knowledge_base,
            "documents",
            (*knowledge_base.documents, Document(content="beta")),
        )
