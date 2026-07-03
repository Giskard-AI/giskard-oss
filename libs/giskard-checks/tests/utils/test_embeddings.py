import sys
from types import SimpleNamespace

from giskard.agents.embeddings import SentenceTransformerEmbedding as AgentsEmbedding
from giskard.checks.utils.embeddings import SentenceTransformerEmbedding


def test_sentence_transformer_embedding_reexport(monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    assert SentenceTransformerEmbedding is AgentsEmbedding
    assert SentenceTransformerEmbedding("all-MiniLM-L6-v2").model == "all-MiniLM-L6-v2"
