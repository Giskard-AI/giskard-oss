from typing import override

import numpy as np
from pydantic import Field
from pydantic.experimental.missing_sentinel import MISSING

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
    reference_text : str | MISSING
        The reference text to compare the output against. If omitted, extracted
        from the trace using ``reference_text_key``.
    reference_text_key : str
        JSONPath expression to extract the reference text from the trace
        (default: "trace.last.metadata.reference_text").

        Can use `trace.last` (preferred) or `trace.interactions[-1]` for JSONPath expressions.
    target_key : str
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
    reference_text: str | MISSING = Field(
        default=MISSING,
        description="The reference text to compare the output with",
    )
    reference_text_key: JSONPathStr = Field(
        default="trace.last.metadata.reference_text",
        description="The key to extract the reference text from the trace",
    )
    target_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description=("The key to extract the actual answer from the trace."),
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
        reference = provided_or_resolve(
            trace,
            key=self.reference_text_key,
            value=self.reference_text,
        )
        if isinstance(reference, NoMatch):
            return CheckResult.error(
                message=f"No value found for reference text key '{self.reference_text_key}'.",
                details={
                    "reference_text_key": self.reference_text_key,
                    "reference_text": reference,
                },
            )
        if reference is None or reference == "":
            return CheckResult.error(
                message="No reference text found",
                details={
                    "reference_text_key": self.reference_text_key,
                    "reference_text": reference,
                },
            )
        actual_answer = resolve(trace, self.target_key)
        if isinstance(actual_answer, NoMatch):
            return CheckResult.error(
                message=f"No value found for actual answer key '{self.target_key}'.",
                details={
                    "actual_answer": actual_answer,
                    "target_key": self.target_key,
                },
            )
        if actual_answer is None or actual_answer == "":
            return CheckResult.error(
                message="No actual answer found",
                details={
                    "actual_answer": actual_answer,
                    "target_key": self.target_key,
                },
            )

        if failure := self._failure_if_collection(
            "reference text", self.reference_text_key, reference
        ):
            return failure
        if failure := self._failure_if_collection(
            "actual answer", self.target_key, actual_answer
        ):
            return failure

        actual_answer = str(actual_answer)
        reference = str(reference)

        emb_a, emb_b = await self.get_embeddings([actual_answer, reference])
        similarity = cosine_similarity(emb_a, emb_b)

        passed = similarity >= self.threshold

        if passed:
            return CheckResult.success(
                message=f"The cosine similarity with the reference answer is {similarity:.2f} which is greater than the threshold {self.threshold:.2f}",
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "actual_answer": actual_answer,
                    "reference_text": reference,
                },
            )
        else:
            return CheckResult.failure(
                message=f"The cosine similarity with the reference answer is {similarity:.2f} which is less than the threshold {self.threshold:.2f}",
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "actual_answer": actual_answer,
                    "reference_text": reference,
                },
            )

    def _failure_if_collection(
        self, label: str, key: str, value: object
    ) -> CheckResult | None:
        """Fail when a keypath resolved to a collection instead of one scalar.

        Wildcard / multi-match paths become lists; ``str(list)`` would embed
        Python repr text nobody wrote. Non-collection scalars still stringify.
        """
        if not isinstance(value, (list, tuple, set, dict)):
            return None
        return CheckResult.failure(
            message=(
                f"Value for {label} key '{key}' must be a single value, but "
                f"found {type(value).__name__}. Use a key that resolves to "
                "one value."
            ),
            details={
                "reference_text_key": self.reference_text_key,
                "target_key": self.target_key,
                "value": str(value),
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
        return await self._embedding_model.embed(texts)
