from typing import override

import numpy as np
from pydantic import Field

from ..core.check import Check
from ..core.extraction import provided_or_resolve, resolve
from ..core.mixin import WithEmbeddingMixin
from ..core.result import CheckResult
from ..core.trace import Trace


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    vec_a = np.asarray(a)
    vec_b = np.asarray(b)

    dot_product = np.dot(vec_a, vec_b)
    print(vec_a, vec_b)
    print(dot_product)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    print(norm)
    if norm == 0:
        raise ValueError("Cannot calculate cosine similarity for null vectors")

    print(dot_product / norm)
    return float(dot_product / norm)


@Check.register("semantic_similarity")
class SemanticSimilarity[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType], WithEmbeddingMixin
):
    threshold: float = Field(
        default=0.95, description="The threshold for the semantic similarity"
    )
    reference_text: str | None = Field(
        default=None, description="The reference text to compare the output with"
    )
    reference_text_key: str = Field(
        default="trace.last.metadata.reference_text",
        description="The key to extract the reference text from the trace",
    )
    actual_answer_key: str = Field(
        default="trace.last.outputs.response",
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
        reference_text = str(
            provided_or_resolve(self.reference_text, trace, self.reference_text_key)
        )
        if not reference_text:
            return CheckResult.failure(
                message="No reference text found",
                details={
                    "reference_text_key": self.reference_text_key,
                    "reference_text": reference_text,
                },
            )
        actual_answer = str(resolve(trace, self.actual_answer_key))
        if not actual_answer:
            return CheckResult.failure(
                message="No actual answer found",
                details={
                    "actual_answer": actual_answer,
                    "actual_answer_key": self.actual_answer_key,
                },
            )

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
        return await self.embedding_model.embed(texts)
