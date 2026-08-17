from pydantic import Field

from ._base import _BaseModel

# -- Embedding types -----------------------------------------------------------


class EmbeddingData(_BaseModel):
    """A single embedding vector and its position in the request batch."""

    embedding: list[float]
    index: int = 0


class EmbeddingUsage(_BaseModel):
    """Token counts reported by an embedding call."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(_BaseModel):
    """Provider-agnostic result of an embedding call."""

    data: list[EmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: EmbeddingUsage | None = None
