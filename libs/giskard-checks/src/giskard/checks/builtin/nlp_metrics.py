"""NLP metric checks for evaluating text quality properties."""

from importlib import import_module
from typing import Literal, Self, override

from pydantic import Field, model_validator

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult, CheckStatus, Metric
from ..utils.optional_deps import require_optional_dependency

_TEXTSTAT_INSTALL_HINT = (
    "The 'textstat' package is required for the Readability check. "
    "Install it with: pip install 'giskard-checks[readability]'"
)

ReadabilityMetric = Literal[
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "automated_readability_index",
    "coleman_liau_index",
    "dale_chall_readability_score",
]


READABILITY_SCORE_GUIDE: dict[ReadabilityMetric, str] = {
    "flesch_reading_ease": (
        "Usually interpreted on a 0-100 scale where higher is easier to read; "
        "60-70 is commonly considered plain English."
    ),
    "flesch_kincaid_grade": (
        "Approximate US school grade level where lower is easier to read; "
        "8-10 is often appropriate for a general audience."
    ),
    "gunning_fog": (
        "Approximate years of formal education needed to understand the text; "
        "scores below 12 are usually easier for broad audiences."
    ),
    "automated_readability_index": (
        "Approximate US school grade level where lower is easier to read; "
        "8-10 is often appropriate for a general audience."
    ),
    "coleman_liau_index": (
        "Approximate US school grade level where lower is easier to read; "
        "8-10 is often appropriate for a general audience."
    ),
    "dale_chall_readability_score": (
        "Lower scores are easier to read; scores around 6 or below are easier, "
        "7-8 is more difficult, and 9+ is very difficult."
    ),
}


@Check.register("readability")
class Readability[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the readability of output text.

    The check extracts a string from the trace, computes a readability score
    with ``textstat``, and optionally validates that score against configured
    minimum and maximum thresholds.

    Requires the optional ``readability`` extra
    (``pip install 'giskard-checks[readability]'``).
    """

    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text to evaluate.",
    )
    metric: ReadabilityMetric = Field(
        default="flesch_reading_ease",
        description=(
            "Readability metric to compute. Metrics either use a higher-is-easier "
            "0-100 style scale (flesch_reading_ease) or lower-is-easier grade/"
            "difficulty style scales (the other metrics)."
        ),
    )
    min_score: float | None = Field(
        default=None,
        description=(
            "Minimum acceptable readability score. Use this for metrics where "
            "higher scores are better, such as flesch_reading_ease."
        ),
    )
    max_score: float | None = Field(
        default=None,
        description=(
            "Maximum acceptable readability score. Use this for grade or "
            "difficulty metrics where lower scores are easier to read."
        ),
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

    @model_validator(mode="after")
    def _require_textstat(self) -> Self:
        """Ensure the optional textstat dependency is installed."""
        require_optional_dependency("textstat", install_hint=_TEXTSTAT_INSTALL_HINT)
        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the readability check against the provided trace."""
        textstat = import_module("textstat")

        text = resolve(trace, self.key)
        details = {
            "key": self.key,
            "metric": self.metric,
            "score_guide": READABILITY_SCORE_GUIDE[self.metric],
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
                details={**details, "value": str(text)},
            )

        try:
            score_fn = getattr(textstat, self.metric)
            score = float(score_fn(text))
        except Exception as exc:
            return CheckResult(
                status=CheckStatus.ERROR,
                message=f"Failed to compute readability score ({self.metric}): {exc}",
                details={**details, "error": str(exc)},
            )
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
