import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, override

from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult

RetrievalScoringStrategy = Literal["strict"]


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _hits_at_k(
    relevant_ids: Sequence[Any], retrieved_ids: Sequence[Any], k: int | None
) -> list[bool]:
    relevant_set = set(relevant_ids)
    seen: set[Any] = set()
    retrieved_at_k = retrieved_ids if k is None else retrieved_ids[:k]
    hits = []
    for doc_id in retrieved_at_k:
        is_hit = doc_id in relevant_set and doc_id not in seen
        hits.append(is_hit)
        seen.add(doc_id)
    return hits


def _safe_divide(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


class RetrievalQualityCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    ABC, Check[InputType, OutputType, TraceType]
):
    """Base class for strict retrieval-quality checks."""

    relevant_ids_key: JSONPathStr = Field(
        ..., description="JSONPath key for the labelled relevant document IDs."
    )
    retrieved_ids_key: JSONPathStr = Field(
        ..., description="JSONPath key for the retrieved document IDs."
    )
    threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum metric score required for the check to pass.",
    )
    strategy: RetrievalScoringStrategy = Field(
        default="strict",
        description="Relevance matching strategy. Strict exact ID matching is supported.",
    )

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Metric name included in result details."""
        ...

    @abstractmethod
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        """Compute the retrieval metric score."""
        ...

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        relevant_ids = resolve(trace, self.relevant_ids_key)
        retrieved_ids = resolve(trace, self.retrieved_ids_key)

        details = {
            "metric": self.metric_name,
            "threshold": self.threshold,
            "strategy": self.strategy,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
        }

        if isinstance(relevant_ids, NoMatch):
            return CheckResult.failure(
                message=f"No value found for relevant IDs key '{self.relevant_ids_key}'.",
                details=details,
            )
        if isinstance(retrieved_ids, NoMatch):
            return CheckResult.failure(
                message=f"No value found for retrieved IDs key '{self.retrieved_ids_key}'.",
                details=details,
            )

        relevant_list = _as_sequence(relevant_ids)
        retrieved_list = _as_sequence(retrieved_ids)
        score = self._score(relevant_list, retrieved_list)
        details["score"] = score

        if score >= self.threshold:
            return CheckResult.success(
                message=f"{self.metric_name} score {score:.3f} met threshold {self.threshold:.3f}.",
                details=details,
            )

        return CheckResult.failure(
            message=f"{self.metric_name} score {score:.3f} is below threshold {self.threshold:.3f}.",
            details=details,
        )


class RetrievalQualityAtKCheck[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument, reportImplicitAbstractClass]
    RetrievalQualityCheck[InputType, OutputType, TraceType]
):
    k: int = Field(..., gt=0, description="Number of retrieved results to evaluate.")


@Check.register("recall_at_k")
class RecallAtK[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityAtKCheck[InputType, OutputType, TraceType]
):
    """Check that Recall@K meets a minimum threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return f"Recall@{self.k}"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        hits = sum(_hits_at_k(relevant_ids, retrieved_ids, self.k))
        return _safe_divide(float(hits), len(set(relevant_ids)))


@Check.register("precision_at_k")
class PrecisionAtK[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityAtKCheck[InputType, OutputType, TraceType]
):
    """Check that Precision@K meets a minimum threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return f"Precision@{self.k}"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        hits = sum(_hits_at_k(relevant_ids, retrieved_ids, self.k))
        return _safe_divide(float(hits), self.k)


@Check.register("hit_rate_at_k")
class HitRateAtK[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityAtKCheck[InputType, OutputType, TraceType]
):
    """Check that HitRate@K meets a minimum threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return f"HitRate@{self.k}"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        return 1.0 if any(_hits_at_k(relevant_ids, retrieved_ids, self.k)) else 0.0


@Check.register("mrr")
class MRR[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityCheck[InputType, OutputType, TraceType]
):
    """Check that mean reciprocal rank for a single query meets a threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return "MRR"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0


@Check.register("ndcg_at_k")
class NDCGAtK[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityAtKCheck[InputType, OutputType, TraceType]
):
    """Check that binary NDCG@K meets a minimum threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return f"NDCG@{self.k}"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        hits = _hits_at_k(relevant_ids, retrieved_ids, self.k)
        dcg = sum(
            1.0 / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1) if hit
        )
        ideal_hits = min(len(set(relevant_ids)), self.k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0


@Check.register("average_precision")
class AveragePrecision[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    RetrievalQualityCheck[InputType, OutputType, TraceType]
):
    """Check that average precision meets a minimum threshold."""

    @property
    @override
    def metric_name(self) -> str:
        return "AveragePrecision"

    @override
    def _score(self, relevant_ids: list[Any], retrieved_ids: list[Any]) -> float:
        hits = _hits_at_k(relevant_ids, retrieved_ids, None)
        precision_sum = 0.0
        hit_count = 0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                hit_count += 1
                precision_sum += hit_count / rank
        return _safe_divide(precision_sum, len(set(relevant_ids)))
