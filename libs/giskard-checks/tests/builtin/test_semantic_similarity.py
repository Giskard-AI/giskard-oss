from typing import cast

import numpy as np
import pytest
from giskard.agents.embeddings.base import BaseEmbeddingModel, EmbeddingParams
from giskard.checks import (
    Check,
    CheckStatus,
    Interaction,
    SemanticSimilarity,
    Trace,
)
from giskard.checks.builtin.semantic_similarity import cosine_similarity
from giskard.checks.core.extraction import NoMatch


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model that returns predictable embeddings."""

    embeddings: dict[str, list[float]]

    async def _embed(
        self, texts: list[str], params: EmbeddingParams | None = None
    ) -> list[np.ndarray]:
        """Return predefined embeddings based on text content."""
        result = []
        for text in texts:
            if text in self.embeddings:
                result.append(np.array(self.embeddings[text]))
            else:
                # Default: return a simple vector based on text length
                result.append(np.array([float(len(text)), 1.0, 0.5]))
        return result


def serialization_roundtrip[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    similarity: SemanticSimilarity[InputType, OutputType, TraceType],
) -> SemanticSimilarity[InputType, OutputType, TraceType]:
    check = Check.model_validate(similarity.model_dump())
    assert isinstance(check, SemanticSimilarity)
    return cast(SemanticSimilarity[InputType, OutputType, TraceType], check)


def test_cosine_similarity_identical_vectors() -> None:
    """Test cosine similarity with identical vectors."""
    vec = np.array([1.0, 2.0, 3.0])
    similarity = cosine_similarity(vec, vec)
    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Test cosine similarity with orthogonal vectors."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    similarity = cosine_similarity(vec_a, vec_b)
    assert np.isclose(similarity, 0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    """Test cosine similarity with opposite vectors."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([-1.0, -2.0, -3.0])
    similarity = cosine_similarity(vec_a, vec_b)
    assert np.isclose(similarity, -1.0)


def test_cosine_similarity_similar_vectors() -> None:
    """Test cosine similarity with similar but not identical vectors."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([1.1, 2.1, 2.9])
    similarity = cosine_similarity(vec_a, vec_b)
    assert similarity > 0.99  # Very similar


def test_cosine_similarity_null_vector_raises_error() -> None:
    """Test that null vectors raise an error."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([0.0, 0.0, 0.0])
    with pytest.raises(
        ValueError, match="Cannot calculate cosine similarity for null vectors"
    ):
        _ = cosine_similarity(vec_a, vec_b)


async def test_run_returns_success() -> None:
    """Test semantic similarity check passes when similarity is above threshold."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "The Eiffel Tower is in Paris.": [1.0, 0.9, 0.8],
            "Paris is home to the Eiffel Tower.": [0.95, 0.92, 0.85],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.95,
        reference_text="Paris is home to the Eiffel Tower.",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
    )
    result = await check.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS
    assert "similarity" in result.details
    assert result.details["similarity"] > 0.95
    assert result.details["threshold"] == 0.95
    assert "The Eiffel Tower is in Paris." in result.details["actual_answer"]
    assert "Paris is home to the Eiffel Tower." in result.details["reference_text"]


async def test_run_returns_failure() -> None:
    """Test semantic similarity check fails when similarity is below threshold."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "The Eiffel Tower is in Paris.": [1.0, 0.0, 0.0],
            "Tokyo is the capital of Japan.": [0.0, 1.0, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.95,
        reference_text="Tokyo is the capital of Japan.",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "Where is the Eiffel Tower?"},
        outputs={"response": "The Eiffel Tower is in Paris."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.FAIL
    assert "similarity" in result.details
    assert result.details["similarity"] < 0.95
    assert result.details["threshold"] == 0.95


async def test_reference_text_from_trace() -> None:
    """Test that reference text can be extracted from trace."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "AI is artificial intelligence.": [1.0, 0.9, 0.8],
            "Artificial intelligence is AI.": [0.98, 0.92, 0.82],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.90,
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "What is AI?"},
        outputs={"response": "AI is artificial intelligence."},
        metadata={"reference_text": "Artificial intelligence is AI."},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert "Artificial intelligence is AI." in result.details["reference_text"]


async def test_direct_reference_text_priority() -> None:
    """Test that direct reference_text takes priority over trace extraction."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Direct reference": [1.0, 0.0, 0.0],
            "Test answer": [0.95, 0.1, 0.0],
            "Trace reference": [0.0, 1.0, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        reference_text="Direct reference",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Test answer"},
        metadata={"reference_text": "Trace reference"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reference_text"] == "Direct reference"


async def test_custom_actual_answer_key() -> None:
    """Test using custom key to extract actual answer."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Custom answer": [1.0, 0.9, 0.8],
            "Reference": [0.95, 0.92, 0.85],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.90,
        reference_text="Reference",
        actual_answer_key="trace.last.outputs.custom_field",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"custom_field": "Custom answer", "response": "Other answer"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["actual_answer"] == "Custom answer"


async def test_custom_reference_text_key() -> None:
    """Test using custom key to extract reference text."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Answer": [1.0, 0.9, 0.8],
            "Custom reference": [0.95, 0.92, 0.85],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.90,
        reference_text_key="trace.last.metadata.custom_ref",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "Test"},
        outputs={"response": "Answer"},
        metadata={"custom_ref": "Custom reference"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["reference_text"] == "Custom reference"


async def test_threshold_variations() -> None:
    """Test different threshold values."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Text A": [1.0, 0.5, 0.3],
            "Text B": [0.9, 0.6, 0.4],
        }
    )

    # Calculate expected similarity
    vec_a = np.array([1.0, 0.5, 0.3])
    vec_b = np.array([0.9, 0.6, 0.4])
    expected_similarity = cosine_similarity(vec_a, vec_b)

    # Test with low threshold (should pass)
    check_low = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.5,
        reference_text="Text B",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(inputs={}, outputs={"response": "Text A"})
    result = await check_low.run(Trace(interactions=[interaction]))
    assert result.status == CheckStatus.PASS

    # Test with high threshold (should fail if similarity < threshold)
    check_high = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.999,
        reference_text="Text B",
        actual_answer_key="trace.last.outputs.response",
    )
    result = await check_high.run(Trace(interactions=[interaction]))
    if expected_similarity < 0.999:
        assert result.status == CheckStatus.FAIL
    else:
        assert result.status == CheckStatus.PASS


async def test_using_trace_last() -> None:
    """Test that check uses trace.last to get the most recent interaction."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "First answer": [1.0, 0.0, 0.0],
            "Second answer": [0.9, 0.1, 0.0],
            "Reference": [0.85, 0.15, 0.0],
        }
    )
    interaction1 = Interaction(
        inputs={"query": "First"},
        outputs={"response": "First answer"},
    )
    interaction2 = Interaction(
        inputs={"query": "Second"},
        outputs={"response": "Second answer"},
    )
    trace = Trace(interactions=[interaction1, interaction2])

    # Verify trace.last points to the last interaction
    assert trace.last is not None
    assert trace.last == interaction2

    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        reference_text="Reference",
        actual_answer_key="trace.last.outputs.response",
    )
    result = await check.run(trace)

    # Should use the second interaction's answer
    assert result.status == CheckStatus.PASS
    assert "Second answer" in result.details["actual_answer"]


async def test_empty_trace() -> None:
    """Test behavior with empty trace."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "None": [1.0, 0.0, 0.0],
            "Reference": [0.9, 0.1, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        reference_text="Reference",
    )
    result = await check.run(Trace())

    # When trace is empty, resolve returns NoMatch which causes check to fail
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for actual answer key 'trace.last.outputs'." in result.message
    )
    assert isinstance(result.details["actual_answer"], NoMatch)


async def test_serialization_roundtrip() -> None:
    """Test that check can be serialized and deserialized."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Answer": [1.0, 0.9, 0.8],
            "Reference": [0.95, 0.92, 0.85],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.92,
        reference_text="Reference",
        actual_answer_key="trace.last.outputs.response",
    )

    # Serialize and deserialize
    roundtrip_check = serialization_roundtrip(check)
    roundtrip_check.embedding_model = embedding_model

    interaction = Interaction(inputs={}, outputs={"response": "Answer"})
    result = await roundtrip_check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["threshold"] == 0.92


async def test_default_threshold() -> None:
    """Test that default threshold is 0.95."""
    embedding_model = MockEmbeddingModel(embeddings={})
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        reference_text="test",
    )
    assert check.threshold == 0.95


async def test_missing_reference_text_in_trace() -> None:
    """Test behavior when reference text is not found in trace."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Answer": [1.0, 0.0, 0.0],
            "None": [0.0, 1.0, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={},
        outputs={"response": "Answer"},
        # No reference_text in metadata
    )
    result = await check.run(Trace(interactions=[interaction]))

    # When reference is not found, check fails early
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for reference text key 'trace.last.metadata.reference_text'."
        in result.message
    )
    assert isinstance(result.details["reference_text"], NoMatch)
    assert "reference_text_key" in result.details


async def test_missing_actual_answer_in_trace() -> None:
    """Test behavior when actual answer field is not found in trace."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "None": [1.0, 0.0, 0.0],
            "Reference": [0.9, 0.1, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        reference_text="Reference",
        actual_answer_key="trace.last.outputs.nonexistent_field",
    )
    interaction = Interaction(
        inputs={},
        outputs={"response": "Some answer"},
        # actual_answer_key points to nonexistent field
    )
    result = await check.run(Trace(interactions=[interaction]))

    # When answer field is not found, check fails early
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for actual answer key 'trace.last.outputs.nonexistent_field'."
        in result.message
    )
    assert isinstance(result.details["actual_answer"], NoMatch)
    assert "actual_answer_key" in result.details


async def test_both_reference_and_answer_missing() -> None:
    """Test behavior when both reference and answer are not found."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "None": [1.0, 0.0, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        actual_answer_key="trace.last.outputs.missing",
        reference_text_key="trace.last.metadata.missing",
    )
    interaction = Interaction(
        inputs={},
        outputs={"response": "Answer"},
        metadata={"other": "data"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    # Check fails when reference text is missing (checked first)
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for reference text key 'trace.last.metadata.missing'."
        in result.message
    )
    assert isinstance(result.details["reference_text"], NoMatch)


async def test_empty_trace_with_no_direct_reference() -> None:
    """Test behavior with empty trace and no direct reference text."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "None": [1.0, 0.0, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
    )
    result = await check.run(Trace())

    # Check fails when reference text is missing
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for reference text key 'trace.last.metadata.reference_text'."
        in result.message
    )
    assert isinstance(result.details["reference_text"], NoMatch)


async def test_missing_outputs_field_in_interaction() -> None:
    """Test behavior when interaction has empty outputs."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "None": [1.0, 0.0, 0.0],
            "Reference": [0.9, 0.1, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        reference_text="Reference",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(
        inputs={"query": "test"},
        outputs={},  # Empty outputs, no response field
    )
    result = await check.run(Trace(interactions=[interaction]))

    # When response field doesn't exist in outputs, check fails
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for actual answer key 'trace.last.outputs.response'."
        in result.message
    )
    assert isinstance(result.details["actual_answer"], NoMatch)


async def test_missing_metadata_in_interaction() -> None:
    """Test behavior when interaction has no metadata."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Answer": [1.0, 0.0, 0.0],
            "None": [0.9, 0.1, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        actual_answer_key="trace.last.outputs.response",
        reference_text_key="trace.last.metadata.reference",
    )
    interaction = Interaction(
        inputs={},
        outputs={"response": "Answer"},
        # No metadata field
    )
    result = await check.run(Trace(interactions=[interaction]))

    # When metadata doesn't exist, check fails (reference not found)
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for reference text key 'trace.last.metadata.reference'."
        in result.message
    )
    assert isinstance(result.details["reference_text"], NoMatch)


async def test_invalid_jsonpath_key() -> None:
    """Test behavior with an invalid JSONPath expression."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Answer": [1.0, 0.0, 0.0],
            "None": [0.9, 0.1, 0.0],
        }
    )
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=0.85,
        actual_answer_key="trace.last.outputs.response",
        reference_text_key="trace.nonexistent.deeply.nested.field",
    )
    interaction = Interaction(
        inputs={},
        outputs={"response": "Answer"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    # Invalid path resolves to NoMatch, causing check to fail
    assert result.status == CheckStatus.FAIL
    assert result.message is not None
    assert (
        "No value found for reference text key 'trace.nonexistent.deeply.nested.field'."
        in result.message
    )
    assert isinstance(result.details["reference_text"], NoMatch)


async def test_similarity_at_exact_threshold() -> None:
    """Test behavior when similarity equals threshold exactly."""
    embedding_model = MockEmbeddingModel(
        embeddings={
            "Text A": [1.0, 0.0, 0.0],
            "Text B": [1.0, 0.0, 0.0],
        }
    )
    # Identical vectors have similarity 1.0
    check = SemanticSimilarity(
        embedding_model=embedding_model,
        threshold=1.0,
        reference_text="Text B",
        actual_answer_key="trace.last.outputs.response",
    )
    interaction = Interaction(inputs={}, outputs={"response": "Text A"})
    result = await check.run(Trace(interactions=[interaction]))

    # Should pass when similarity >= threshold
    assert result.status == CheckStatus.PASS
    assert result.details["similarity"] == 1.0
