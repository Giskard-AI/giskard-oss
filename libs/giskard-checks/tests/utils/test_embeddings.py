import sys
import types

import numpy as np
import pytest
from giskard.agents.embeddings.base import BaseEmbeddingModel
from giskard.checks import CheckStatus, Interaction, SemanticSimilarity, Trace
from giskard.checks.utils import optional_deps
from giskard.checks.utils.embeddings import SentenceTransformerEmbedding


class FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert normalize_embeddings is False

        vectors: list[np.ndarray] = []
        for text in texts:
            if text == "Hello world":
                vectors.append(np.array([1.0, 0.0, 0.0]))
            elif text == "Hello there":
                vectors.append(np.array([0.95, 0.05, 0.0]))
            else:
                vectors.append(np.array([0.0, 1.0, 0.0]))
        return np.vstack(vectors)


def test_sentence_transformer_embedding_is_registered() -> None:
    model = BaseEmbeddingModel.model_validate(
        {"kind": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"}
    )

    assert isinstance(model, SentenceTransformerEmbedding)
    assert model.model_name == "all-MiniLM-L6-v2"


async def test_sentence_transformer_embedding_produces_valid_similarity_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    check = SemanticSimilarity(
        embedding_model=SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"),
        threshold=0.9,
        reference_text="Hello there",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(inputs={}, outputs={"response": "Hello world"})

    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["similarity"] > 0.9


async def test_sentence_transformer_embedding_raises_helpful_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    real_import_module = optional_deps.importlib.import_module

    def fake_import_module(name: str, /, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(optional_deps.importlib, "import_module", fake_import_module)

    model = SentenceTransformerEmbedding()

    with pytest.raises(ValueError, match="local-embeddings"):
        await model.embed(["Hello world"])
