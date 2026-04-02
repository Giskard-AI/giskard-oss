"""NLP metric checks for evaluating text quality properties."""

from typing import Literal, override

from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult, CheckStatus, Metric

TEXTSTAT_METRICS = Literal["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"]


@Check.register("readability")
class Readability[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the readability of model output text.

        Uses the ``textstat`` library to compute a readability score for the
            extracted text and validates it against optional min/max thresholds.

                Supports three readability metrics:

                    - ``flesch_reading_ease``: Higher scores (0-100) indicate easier reading.
                          Scores above 60 are considered "standard" / easy to read.
                              - ``flesch_kincaid_grade``: Returns the US school grade level required
                                    to understand the text. Lower values = simpler text.
                                        - ``gunning_fog``: Estimates the years of formal education needed to
                                              understand the text on a first reading. Lower values = simpler text.

                                                  Attributes
                                                      ----------
                                                          key : JSONPathStr
                                                                  JSONPath expression to extract the text from the trace
                                                                          (default: ``"trace.last.outputs"``).
                                                                              metric : str
                                                                                      The readability metric to compute. One of ``"flesch_reading_ease"``,
                                                                                              ``"flesch_kincaid_grade"``, or ``"gunning_fog"``
                                                                                                      (default: ``"flesch_reading_ease"``).
                                                                                                          min_score : float or None
                                                                                                                  Minimum acceptable score. If ``None``, no lower bound is enforced.
                                                                                                                      max_score : float or None
                                                                                                                              Maximum acceptable score. If ``None``, no upper bound is enforced.
                                                                                                                              
                                                                                                                                  Examples
                                                                                                                                      --------
                                                                                                                                          >>> from giskard.checks.builtin.nlp_metrics import Readability
                                                                                                                                              >>> from giskard.checks import Trace, Interaction
                                                                                                                                                  >>> import asyncio
                                                                                                                                                      >>> check = Readability(metric="flesch_reading_ease", min_score=0)
                                                                                                                                                          >>> interaction = Interaction(inputs="Tell me about dogs", outputs="Dogs are friendly pets.")
                                                                                                                                                              >>> result = asyncio.run(check.run(Trace(interactions=[interaction])))
                                                                                                                                                                  >>> result.passed
                                                                                                                                                                      True
                                                                                                                                                                          """

    key: JSONPathStr = Field(
              default="trace.last.outputs",
              description="JSONPath expression to extract the text to evaluate.",
    )
    metric: TEXTSTAT_METRICS = Field(
              default="flesch_reading_ease",
              description=(
                            "Readability metric to compute. One of 'flesch_reading_ease', "
                            "'flesch_kincaid_grade', or 'gunning_fog'."
              ),
    )
    min_score: float | None = Field(
              default=None,
              description="Minimum acceptable readability score. None means no lower bound.",
    )
    max_score: float | None = Field(
              default=None,
              description="Maximum acceptable readability score. None means no upper bound.",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
              """Execute the readability check against the provided trace.

                      Parameters
                              ----------
                                      trace : TraceType
                                                  The trace containing interaction history.

                                                          Returns
                                                                  -------
                                                                          CheckResult
                                                                                      The result of the check evaluation.
                                                                                              """
              try:
                            import textstat
except ImportError:
            return CheckResult(
                              status=CheckStatus.ERROR,
                              message=(
                                                    "The 'textstat' package is required for the Readability check. "
                                                    "Install it with: pip install 'giskard-checks[nlp]'"
                              ),
            )

        text = resolve(trace, self.key)

        if isinstance(text, NoMatch):
                      return CheckResult.failure(
                                        message=f"No value found for key '{self.key}'.",
                                        details={"key": self.key, "text": text},
                      )

        text = str(text)

        score_fn = getattr(textstat, self.metric)
        score: float = float(score_fn(text))

        details = {
                      "text": text,
                      "metric": self.metric,
                      "score": score,
                      "min_score": self.min_score,
                      "max_score": self.max_score,
        }
        metrics = [Metric(name=self.metric, value=score)]

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
