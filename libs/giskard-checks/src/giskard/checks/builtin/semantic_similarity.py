from typing import override

import numpy as np
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve, resolve
from ..core.mixin import WithEmbeddingMixin
from ..core.result import CheckResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector for comparison.
    b : np.ndarray
        Second vector for comparison.

    Returns
    -------
    float
        Cosine similarity score between -1 and 1, where 1 indicates identical
        vectors and 0 indicates orthogonal vectors.

    Raises
    ------
    ValueError
        If either vector is a null vector (zero norm).
    """
    vec_a = np.asarray(a)
    vec_b = np.asarray(b)

    dot_product = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        raise ValueError("Cannot calculate cosine similarity for null vectors")

    return float(dot_product / norm)


@Check.register("semantic_similarity")
class SemanticSimilarity[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], WithEmbeddingMixin
):
    """Check that validates semantic similarity between outputs and reference text.

    Uses embeddings to compute cosine similarity between the actual answer and
    a reference text. The check passes if the similarity score meets or exceeds
    the specified threshold.

    Attributes
    ----------
    threshold : float
        The minimum cosine similarity score required for the check to pass
        (default: 0.95).
    reference_text : str | None
        The reference text to compare the output against. If None, the reference
        text will be extracted from the trace using `reference_text_key`.
    reference_text_key : str
        JSONPath expression to extract the reference text from the trace
        (default: "trace.last.metadata.reference_text").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    actual_answer_key : str
        JSONPath expression to extract the actual answer from the trace
        (default: "trace.last.outputs").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    embedding_model : BaseEmbeddingModel
        Embedding model for generating vector representations (inherited from WithEmbeddingMixin).

    Examples
    --------
    >>> from giskard.checks import SemanticSimilarity
    >>> check = SemanticSimilarity(
    ...     name="answer_similarity",
    ...     reference_text="The capital of France is Paris",
    ...     threshold=0.95
    ... )
    """

    threshold: float = Field(
        default=0.95, description="The threshold for the semantic similarity"
    )
    reference_text: str | None = Field(
        default=None, description="The reference text to compare the output with"
    )
    reference_text_key: JSONPathStr = Field(
        default="trace.last.metadata.reference_text",
        description="The key to extract the reference text from the trace",
    )
    actual_answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="The key to extract the actual answer from the trace",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the semantic similarity check.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history. Access the current
            interaction via `trace.last` (preferred in prompt templates) or
            `trace.interactions[-1]` if available.

        Returns
        -------
        CheckResult
            The result of the check evaluation.
        """
        reference_text = provided_or_resolve(
            trace,
            key=provide_not_none(self.reference_text_key),
            value=provide_not_none(self.reference_text),
        )
        if isinstance(reference_text, NoMatch):
            return CheckResult.failure(
                message=f"No value found for reference text key '{self.reference_text_key}'.",
                details={
                    "reference_text_key": self.reference_text_key,
                    "reference_text": reference_text,
                },
            )
        if reference_text is None or reference_text == "":
            return CheckResult.failure(
                message="No reference text found",
                details={
                    "reference_text_key": self.reference_text_key,
                    "reference_text": reference_text,
                },
            )
        actual_answer = resolve(trace, self.actual_answer_key)
        if isinstance(actual_answer, NoMatch):
            return CheckResult.failure(
                message=f"No value found for actual answer key '{self.actual_answer_key}'.",
                details={
                    "actual_answer": actual_answer,
                    "actual_answer_key": self.actual_answer_key,
                },
            )
        if actual_answer is None or actual_answer == "":
            return CheckResult.failure(
                message="No actual answer found",
                details={
                    "actual_answer": actual_answer,
                    "actual_answer_key": self.actual_answer_key,
                },
            )

        actual_answer = str(actual_answer)
        reference_text = str(reference_text)

        emb_a, emb_b = await self.get_embeddings([actual_answer, reference_text])
        similarity = cosine_similarity(emb_a, emb_b)

        passed = similarity >= self.threshold

        if passed:
            return CheckResult.success(
                message=f"The cosine similarity with the reference answer is {similarity:.2f} which is greater than the threshold {self.threshold:.2f}",
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "actual_answer": actual_answer,
                    "reference_text": reference_text,
                },
            )
        else:
            return CheckResult.failure(
                message=f"The cosine similarity with the reference answer is {similarity:.2f} which is less than the threshold {self.threshold:.2f}",
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "actual_answer": actual_answer,
                    "reference_text": reference_text,
                },
            )

    async def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for the given texts.

        Parameters
        ----------
        texts : list[str]
            List of text strings to generate embeddings for.

        Returns
        -------
        list[np.ndarray]
            List of embedding vectors, one for each input text.
        """
        return await self.embedding_model.embed(texts)
