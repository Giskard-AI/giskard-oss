from pydantic import Field

from ._base import _BaseModel

# -- Embedding types -----------------------------------------------------------


class EmbeddingData(_BaseModel):
    embedding: list[float]
    index: int = 0


class EmbeddingUsage(_BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(_BaseModel):
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: EmbeddingUsage | None = None
