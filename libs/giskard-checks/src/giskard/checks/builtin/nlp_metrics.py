"""NLP-based metric checks.

This module provides lightweight, locally-computed checks that do not require
an LLM API call:

- :class:`Sentiment` — validates the sentiment polarity of text against an
  expected label or a numeric score range, using the ``textblob`` library.
"""

from typing import Literal, override

from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve
from ..core.result import CheckResult, CheckStatus, Metric

_SENTIMENT_LABELS = Literal["positive", "negative", "neutral"]

_TEXTBLOB_MISSING_MSG = (
    "The 'textblob' package is required for the Sentiment check but is not "
    "installed. Install it with: pip install 'giskard-checks[nlp]'"
)


def _polarity_to_label(polarity: float) -> _SENTIMENT_LABELS:
    """Convert a TextBlob polarity score to a human-readable label.

    Parameters
    ----------
    polarity : float
        Polarity score in the range [-1.0, 1.0].

    Returns
    -------
    _SENTIMENT_LABELS
        ``"positive"`` if polarity > 0.05, ``"negative"`` if polarity < -0.05,
        ``"neutral"`` otherwise.
    """
    if polarity > 0.05:
        return "positive"
    if polarity < -0.05:
        return "negative"
    return "neutral"


@Check.register("sentiment")
class Sentiment[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the sentiment polarity of model output text.

    Uses the :mod:`textblob` library to compute a polarity score in
    ``[-1.0, 1.0]`` without making any LLM API calls.  The check can
    assert:

    * an **expected sentiment label** (``"positive"``, ``"negative"``, or
      ``"neutral"``), and/or
    * a **numeric score range** via ``min_score`` / ``max_score``.

    Both the polarity score and the derived sentiment label are attached to the
    result as :class:`~giskard.checks.core.result.Metric` values.

    Attributes
    ----------
    text : str | None
        The text to analyse.  If ``None``, the text is extracted from the trace
        using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression to extract the text from the trace (default:
        ``"trace.last.outputs"``).

        Can use ``trace.last`` (preferred) or ``trace.interactions[-1]`` for
        JSONPath expressions.
    expected : Literal["positive", "negative", "neutral"] | None
        Expected sentiment label.  The check fails if the computed label does
        not match.  If ``None``, no label assertion is performed.
    min_score : float | None
        Minimum acceptable polarity score (inclusive).  ``None`` means no lower
        bound.
    max_score : float | None
        Maximum acceptable polarity score (inclusive).  ``None`` means no upper
        bound.

    Examples
    --------
    Assert the response is positive:

    >>> from giskard.checks import Sentiment, Scenario
    >>> scenario = (
    ...     Scenario(name="positive_response")
    ...     .interact(inputs="How is our service?", outputs="Your service is excellent!")
    ...     .check(Sentiment(expected="positive"))
    ... )

    Assert polarity is above a numeric threshold:

    >>> check = Sentiment(
    ...     text="This product is fantastic and I love it!",
    ...     min_score=0.3,
    ... )

    Combine label and score constraints:

    >>> check = Sentiment(
    ...     text="Terrible experience, very disappointed.",
    ...     expected="negative",
    ...     max_score=-0.1,
    ... )
    """

    text: str | None = Field(
        default=None,
        description=(
            "The text to analyse for sentiment. If None, extracted from the "
            "trace using text_key."
        ),
    )
    text_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text from the trace.",
    )
    expected: _SENTIMENT_LABELS | None = Field(
        default=None,
        description=(
            "Expected sentiment label: 'positive', 'negative', or 'neutral'. "
            "If None, no label assertion is performed."
        ),
    )
    min_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Minimum acceptable polarity score (inclusive, range -1.0 to 1.0).",
    )
    max_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Maximum acceptable polarity score (inclusive, range -1.0 to 1.0).",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the sentiment check.

        Parameters
        ----------
        trace : TraceType
            The trace containing interaction history. Used to extract text if
            not provided directly.

        Returns
        -------
        CheckResult
            * ``PASS`` — all asserted constraints are satisfied.
            * ``FAIL`` — one or more constraints are violated.
            * ``ERROR`` — ``textblob`` is not installed, or text could not be
              extracted.
        """
        try:
            from textblob import TextBlob  # type: ignore[import-untyped]
        except ImportError:
            return CheckResult.error(
                message=_TEXTBLOB_MISSING_MSG,
                details={"error": "textblob_not_installed"},
            )

        # --- Extract text ---------------------------------------------------
        raw = provided_or_resolve(
            trace,
            key=self.text_key,
            value=provide_not_none(self.text),
        )

        if isinstance(raw, NoMatch):
            return CheckResult.error(
                message=f"No value found for text key '{self.text_key}'.",
                details={"text_key": self.text_key},
            )

        text = str(raw)

        # --- Compute sentiment ----------------------------------------------
        polarity: float = TextBlob(text).sentiment.polarity  # type: ignore[attr-defined]
        label = _polarity_to_label(polarity)

        metrics = [
            Metric(name="polarity_score", value=round(polarity, 6)),
        ]
        details: dict = {
            "text": text,
            "polarity_score": polarity,
            "sentiment_label": label,
            "expected": self.expected,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }

        # --- Evaluate constraints -------------------------------------------
        failures: list[str] = []

        if self.expected is not None and label != self.expected:
            failures.append(
                f"Expected sentiment '{self.expected}' but got '{label}' "
                f"(polarity={polarity:.4f})."
            )

        if self.min_score is not None and polarity < self.min_score:
            failures.append(
                f"Polarity score {polarity:.4f} is below the minimum threshold "
                f"{self.min_score}."
            )

        if self.max_score is not None and polarity > self.max_score:
            failures.append(
                f"Polarity score {polarity:.4f} exceeds the maximum threshold "
                f"{self.max_score}."
            )

        if failures:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=" ".join(failures),
                metrics=metrics,
                details=details,
            )

        success_parts: list[str] = [
            f"Sentiment is '{label}' (polarity={polarity:.4f})."
        ]
        if self.expected is not None:
            success_parts.append(f"Expected label '{self.expected}' matched.")
        if self.min_score is not None or self.max_score is not None:
            low = self.min_score if self.min_score is not None else -1.0
            high = self.max_score if self.max_score is not None else 1.0
            success_parts.append(
                f"Score {polarity:.4f} is within the required range [{low}, {high}]."
            )

        return CheckResult(
            status=CheckStatus.PASS,
            message=" ".join(success_parts),
            metrics=metrics,
            details=details,
        )
