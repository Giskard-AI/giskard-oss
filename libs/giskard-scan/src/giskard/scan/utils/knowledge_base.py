"""Knowledge base primitives for document-grounded scan generators."""

import asyncio
from typing import ClassVar, Self

import numpy as np
from giskard.checks import WithEmbeddingMixin
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator


class Document(BaseModel):
    """Document stored in a scan knowledge base.

    Attributes:
        content: Text content used for question generation and grounding.
        embeddings: Optional embedding vector. Missing vectors are computed
            lazily when nearest-neighbor retrieval is requested.
        tags: Optional document labels carried by the caller.
    """

    content: str
    embeddings: list[float] | None = None
    tags: list[str] | None = None


class KnowledgeBase(WithEmbeddingMixin):
    """Collection of documents used by knowledge-base scenario generators.

    Embeddings are not computed when the knowledge base is created. They are
    filled lazily, in batches, the first time nearest-neighbor retrieval needs
    them. The public document collection is immutable so documents cannot be
    added after embeddings have been computed.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    documents: tuple[Document, ...]
    _embedding_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    # Validated embedding matrix and per-row norms, computed once from the
    # frozen documents and reused across every closest_documents call.
    _matrix_cache: tuple[np.ndarray, np.ndarray] | None = PrivateAttr(default=None)

    @classmethod
    def from_texts(cls, texts: list[str]) -> Self:
        """Create a knowledge base from raw text documents.

        Args:
            texts: Text chunks to wrap as :class:`Document` objects.

        Returns:
            A knowledge base containing one document per input text.
        """
        return cls(documents=tuple(Document(content=text) for text in texts))

    @field_validator("documents", mode="after")
    @classmethod
    def _validate_documents(
        cls, documents: tuple[Document, ...]
    ) -> tuple[Document, ...]:
        non_empty_documents = tuple(doc for doc in documents if doc.content.strip())
        if not non_empty_documents:
            raise ValueError(
                "KnowledgeBase must contain at least one non-empty document"
            )
        return non_empty_documents

    async def ensure_embeddings(self) -> None:
        """Ensure every document has embeddings from the same model.

        If any document is missing embeddings, all embeddings are recomputed in
        one batch. This avoids mixing vectors that may have been produced by
        different embedding models.
        """
        if all(doc.embeddings is not None for doc in self.documents):
            return

        async with self._embedding_lock:
            if all(doc.embeddings is not None for doc in self.documents):
                return

            embeddings = await self._embedding_model.embed(
                [document.content for document in self.documents]
            )
            if len(embeddings) != len(self.documents):
                raise ValueError(
                    "Embedding model returned a different number of vectors than documents"
                )

            for document, embedding in zip(self.documents, embeddings):
                document.embeddings = np.asarray(embedding, dtype=float).tolist()

    async def closest_documents(
        self, seed_index: int, max_documents: int
    ) -> list[Document]:
        """Return the documents closest to a seed document by cosine similarity.

        Args:
            seed_index: Index of the seed document in ``documents``.
            max_documents: Maximum number of documents to return, including the
                seed document itself.

        Returns:
            Documents sorted from highest to lowest cosine similarity.
        """
        if not 0 <= seed_index < len(self.documents):
            raise IndexError(f"seed_index out of range: {seed_index}")
        if max_documents <= 0:
            return []

        await self.ensure_embeddings()
        matrix, row_norms = self._embedding_matrix()
        similarities = (matrix @ matrix[seed_index]) / (
            row_norms * row_norms[seed_index]
        )
        indices = np.argsort(-similarities)[:max_documents]
        return [self.documents[int(index)] for index in indices]

    def _embedding_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix_cache is not None:
            return self._matrix_cache

        embeddings = [doc.embeddings for doc in self.documents]
        if any(embedding is None for embedding in embeddings):
            raise ValueError("KnowledgeBase embeddings are incomplete")

        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("KnowledgeBase embeddings must be a 2D matrix")
        # NaN/Inf must be caught before the norm check: norm(nan_vector) is nan,
        # and nan == 0 is False, so non-finite vectors slip past the zero-vector
        # guard and corrupt cosine-similarity ordering with no error.
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "KnowledgeBase embeddings must not contain non-finite values"
            )
        row_norms = np.linalg.norm(matrix, axis=1)
        if np.any(row_norms == 0):
            raise ValueError("KnowledgeBase embeddings must not contain zero vectors")

        self._matrix_cache = (matrix, row_norms)
        return self._matrix_cache


def normalize_knowledge_base(
    knowledge_base: KnowledgeBase | list[str] | None,
) -> KnowledgeBase | None:
    """Normalize supported knowledge base inputs.

    Args:
        knowledge_base: Either an existing knowledge base, raw text documents,
            or ``None``.

    Returns:
        A :class:`KnowledgeBase` instance, or ``None`` when no input was
        provided.
    """
    if knowledge_base is None or isinstance(knowledge_base, KnowledgeBase):
        return knowledge_base
    # A bare str is a valid Iterable[str], so from_texts would silently treat it
    # as one document per character. Reject it before that footgun fires.
    if isinstance(knowledge_base, str):
        raise TypeError(
            "knowledge_base must be a list of strings or a KnowledgeBase, "
            "not a single string"
        )
    return KnowledgeBase.from_texts(knowledge_base)
