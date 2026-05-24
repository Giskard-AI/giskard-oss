"""NLP-based metric checks.

Lightweight, locally-computed checks that do not require an LLM API call.

- :class:`Sentiment` — validates the sentiment polarity of text against an
  expected label or a numeric polarity range, using the ``textblob`` library.
"""

from typing import Literal, override

from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.exceptions import require_optional
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve
from ..core.result import CheckResult, CheckStatus, Metric

_SENTIMENT_LABELS = Literal["positive", "negative", "neutral"]
_POLARITY_LABEL_THRESHOLD = 0.05


def _polarity_to_label(polarity: float) -> _SENTIMENT_LABELS:
    """Map a TextBlob polarity score to a sentiment label.

    Parameters
    ----------
    polarity : float
        Polarity score in the closed range ``[-1.0, 1.0]``.

    Returns
    -------
    _SENTIMENT_LABELS
        ``"positive"`` if polarity > 0.05, ``"negative"`` if polarity < -0.05,
        ``"neutral"`` otherwise.
    """
    if polarity > _POLARITY_LABEL_THRESHOLD:
        return "positive"
    if polarity < -_POLARITY_LABEL_THRESHOLD:
        return "negative"
    return "neutral"


@Check.register("sentiment")
class Sentiment[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the sentiment of the output text.

    Uses the :mod:`textblob` library to compute a polarity score in the
    closed range ``[-1.0, 1.0]`` without any LLM API call. The check can
    assert an expected sentiment label, a polarity-score range, or both.

    Requires the ``nlp`` extra: ``pip install 'giskard-checks[nlp]'``.

    Attributes
    ----------
    text : str | None
        The text to analyse. If ``None``, the text is extracted from the
        trace using ``text_key``.
    text_key : JSONPathStr
        JSONPath expression used to extract the text from the trace.
        Defaults to ``"trace.last.outputs"``.
    expected : Literal["positive", "negative", "neutral"] | None
        Expected sentiment label. The check fails if the computed label
        differs. ``None`` skips the label assertion.
    min_score : float | None
        Minimum acceptable polarity score (inclusive). ``None`` skips the
        lower-bound assertion.
    max_score : float | None
        Maximum acceptable polarity score (inclusive). ``None`` skips the
        upper-bound assertion.

    Examples
    --------
    Assert that the output is positive:

    >>> from giskard.checks import Sentiment
    >>> check = Sentiment(expected="positive")

    Assert that the polarity score stays above a threshold:

    >>> check = Sentiment(min_score=0.3)
    """

    text: str | None = Field(
        default=None,
        description="The text to analyse. If None, extracted from trace via text_key.",
    )
    text_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the text from the trace.",
    )
    expected: _SENTIMENT_LABELS | None = Field(
        default=None,
        description=(
            "Expected sentiment label: 'positive', 'negative' or 'neutral'. "
            "If None, no label assertion is performed."
        ),
    )
    min_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Minimum acceptable polarity score (inclusive, in [-1.0, 1.0]).",
    )
    max_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Maximum acceptable polarity score (inclusive, in [-1.0, 1.0]).",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Compute sentiment and validate it against the configured assertions.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history.

        Returns
        -------
        CheckResult
            Result with status, message, the polarity as a metric, and
            the sentiment label plus inputs in ``details``.
        """
        text = provided_or_resolve(
            trace, key=self.text_key, value=provide_not_none(self.text)
        )
        if isinstance(text, NoMatch):
            return CheckResult.failure(
                message=f"No value found for text key '{self.text_key}'.",
                details={"text_key": self.text_key},
            )
        if text is None or text == "":
            return CheckResult.failure(
                message="No text found to analyse.",
                details={"text_key": self.text_key, "text": text},
            )
        if not isinstance(text, str):
            return CheckResult.failure(
                message=(
                    f"Value for text is not a string, expected string but got "
                    f"{type(text).__name__}."
                ),
                details={"text_key": self.text_key, "text": text},
            )

        textblob = require_optional("textblob", "nlp", feature="the Sentiment check")

        polarity = float(textblob.TextBlob(text).sentiment.polarity)
        label = _polarity_to_label(polarity)

        details = {
            "text": text,
            "sentiment_label": label,
            "polarity": polarity,
            "expected": self.expected,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }
        metrics = [Metric(name="sentiment_polarity", value=polarity)]

        if self.expected is not None and label != self.expected:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=(
                    f"Sentiment is '{label}' (polarity {polarity:.4f}) but "
                    f"expected '{self.expected}'."
                ),
                details=details,
                metrics=metrics,
            )

        if self.min_score is not None or self.max_score is not None:
            low = self.min_score if self.min_score is not None else -1.0
            high = self.max_score if self.max_score is not None else 1.0
            if polarity < low or polarity > high:
                return CheckResult(
                    status=CheckStatus.FAIL,
                    message=(
                        f"Polarity {polarity:.4f} is outside the required "
                        f"range [{low}, {high}]."
                    ),
                    details=details,
                    metrics=metrics,
                )

        return CheckResult(
            status=CheckStatus.PASS,
            message=f"Sentiment is '{label}' (polarity {polarity:.4f}).",
            details=details,
            metrics=metrics,
        )
