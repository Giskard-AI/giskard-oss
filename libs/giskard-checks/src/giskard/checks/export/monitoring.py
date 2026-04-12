import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.result import SuiteResult
from ..scenarios.suite import Suite


class ResultExporter(Protocol):
    def export(self, result: SuiteResult) -> None: ...


@dataclass(frozen=True)
class ProductionEvaluationAlert:
    pass_rate: float
    threshold: float
    message: str


@dataclass(frozen=True)
class ProductionEvaluationResult:
    result: SuiteResult | None
    alert: ProductionEvaluationAlert | None
    sampled: bool


async def evaluate_production_sample(
    suite: Suite,
    *,
    target: Any | None = None,
    exporters: Iterable[ResultExporter] = (),
    sample_rate: float = 1.0,
    min_pass_rate: float | None = None,
) -> ProductionEvaluationResult:
    """Run checks for a sampled production request and export the result."""
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0.0 and 1.0")
    if min_pass_rate is not None and not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0.0 and 1.0")

    if random.random() > sample_rate:
        return ProductionEvaluationResult(result=None, alert=None, sampled=False)

    result = await suite.run(target=target)
    for exporter in exporters:
        exporter.export(result)

    alert = None
    if min_pass_rate is not None and result.pass_rate < min_pass_rate:
        alert = ProductionEvaluationAlert(
            pass_rate=result.pass_rate,
            threshold=min_pass_rate,
            message=(
                f"Giskard pass rate {result.pass_rate:.3f} is below "
                f"threshold {min_pass_rate:.3f}."
            ),
        )

    return ProductionEvaluationResult(result=result, alert=alert, sampled=True)
