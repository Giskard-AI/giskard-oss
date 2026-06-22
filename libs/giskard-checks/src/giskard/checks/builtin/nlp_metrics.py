"""NLP metric check implementations."""

from importlib import import_module
from typing import Any, Literal, Protocol, Self, cast, override

from pydantic import Field, model_validator

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult, CheckStatus, Metric

ReadabilityMetric = Literal[
    "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"
]


class TextstatModule(Protocol):
    def flesch_reading_ease(self, text: str) -> float: ...

    def flesch_kincaid_grade(self, text: str) -> float: ...

    def gunning_fog(self, text: str) -> float: ...


def _load_textstat() -> TextstatModule:
    try:
        return cast(TextstatModule, cast(object, import_module("textstat")))
    except ModuleNotFoundError as err:
        if err.name != "textstat":
            raise
        raise RuntimeError(
            "The textstat package is required to use Readability. "
            "Install it with `giskard-checks[nlp]`."
        ) from err


@Check.register("readability")
class Readability[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates readability metrics for text outputs."""

    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text to score.",
    )
    metric: ReadabilityMetric = Field(
        default="flesch_reading_ease",
        description="Readability metric to compute.",
    )
    min_score: float | None = Field(
        default=None,
        description="Optional minimum accepted readability score.",
    )
    max_score: float | None = Field(
        default=None,
        description="Optional maximum accepted readability score.",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.min_score > self.max_score
        ):
            raise ValueError("min_score must be less than or equal to max_score.")
        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        text = resolve(trace, self.key)
        details: dict[str, Any] = {
            "key": self.key,
            "text": text,
            "metric": self.metric,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

        if isinstance(text, NoMatch):
            return CheckResult.failure(
                message=f"No value found for key '{self.key}'.",
                details=details,
            )

        if not isinstance(text, str):
            return CheckResult.failure(
                message=f"Value for key '{self.key}' is not a string, expected string but got {type(text).__name__}.",
                details=details,
            )

        try:
            score = self._compute_score(text)
        except RuntimeError as err:
            details["error"] = str(err)
            return CheckResult.error(message=str(err), details=details)

        details["score"] = score
        metric = Metric(name=self.metric, value=score)
        failures = self._threshold_failures(score)

        if failures:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=" ".join(failures),
                metrics=[metric],
                details=details,
            )

        return CheckResult(
            status=CheckStatus.PASS,
            message=f"Readability score {score:.2f} satisfies configured thresholds.",
            metrics=[metric],
            details=details,
        )

    def _compute_score(self, text: str) -> float:
        textstat = _load_textstat()
        scorer = getattr(textstat, self.metric)
        return float(scorer(text))

    def _threshold_failures(self, score: float) -> list[str]:
        failures = []
        if self.min_score is not None and score < self.min_score:
            failures.append(
                f"Readability score {score:.2f} is below the minimum {self.min_score:.2f}."
            )
        if self.max_score is not None and score > self.max_score:
            failures.append(
                f"Readability score {score:.2f} is above the maximum {self.max_score:.2f}."
            )
        return failures
