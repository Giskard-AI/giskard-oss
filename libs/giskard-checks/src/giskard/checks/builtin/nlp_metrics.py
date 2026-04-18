"""NLP metric checks for evaluating text quality properties."""

from importlib import import_module
from typing import Literal, Self, override

from pydantic import Field, model_validator

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult, CheckStatus, Metric

ReadabilityMetric = Literal[
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
]


@Check.register("readability")
class Readability[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the readability of output text.

    The check extracts a string from the trace, computes a readability score
    with ``textstat``, and optionally validates that score against configured
    minimum and maximum thresholds.
    """

    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text to evaluate.",
    )
    metric: ReadabilityMetric = Field(
        default="flesch_reading_ease",
        description="Readability metric to compute.",
    )
    min_score: float | None = Field(
        default=None,
        description="Minimum acceptable readability score.",
    )
    max_score: float | None = Field(
        default=None,
        description="Maximum acceptable readability score.",
    )

    @model_validator(mode="after")
    def validate_score_range(self) -> Self:
        """Ensure the optional thresholds define a valid interval."""
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.min_score > self.max_score
        ):
            raise ValueError("min_score must be less than or equal to max_score")
        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the readability check against the provided trace."""
        try:
            textstat = import_module("textstat")
        except ImportError:
            return CheckResult(
                status=CheckStatus.ERROR,
                message=(
                    "The 'textstat' package is required for the Readability check. "
                    "Install it with: pip install 'giskard-checks[nlp]'"
                ),
                details={"package": "textstat"},
            )

        text = resolve(trace, self.key)
        details = {
            "key": self.key,
            "metric": self.metric,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

        if isinstance(text, NoMatch):
            return CheckResult.failure(
                message=f"No value found for key '{self.key}'.",
                details={**details, "text": text},
            )

        if not isinstance(text, str):
            return CheckResult.failure(
                message=(
                    f"Value for key '{self.key}' must be a string, but found "
                    f"{type(text).__name__}."
                ),
                details={**details, "value": text},
            )

        score_fn = getattr(textstat, self.metric)
        score = float(score_fn(text))
        metrics = [Metric(name=self.metric, value=score)]
        details = {**details, "text": text, "score": score}

        if self.min_score is not None and score < self.min_score:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=(
                    f"Readability score {score:.2f} ({self.metric}) is below "
                    f"the minimum threshold of {self.min_score}."
                ),
                metrics=metrics,
                details=details,
            )

        if self.max_score is not None and score > self.max_score:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=(
                    f"Readability score {score:.2f} ({self.metric}) exceeds "
                    f"the maximum threshold of {self.max_score}."
                ),
                metrics=metrics,
                details=details,
            )

        return CheckResult(
            status=CheckStatus.PASS,
            message=(
                f"Readability score {score:.2f} ({self.metric}) is within "
                "the acceptable range."
            ),
            metrics=metrics,
            details=details,
        )
