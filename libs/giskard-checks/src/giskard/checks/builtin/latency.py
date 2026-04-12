from typing import override

from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult, CheckStatus, Metric


@Check.register("latency")
class Latency[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates the response latency is within an acceptable threshold.

    This check reads latency (in milliseconds) from the interaction metadata and
    asserts it does not exceed a configurable maximum. The actual latency is also
    reported as a ``Metric`` for observability.

    To use this check, store the response time in the interaction metadata under
    the key referenced by ``key`` (default ``trace.last.metadata.latency_ms``):

    .. code-block:: python

        Interaction(
            inputs="Hello",
            outputs="Hi there!",
            metadata={"latency_ms": 312},
        )

    Attributes
    ----------
    max_seconds : float
        Maximum allowed response time in seconds. Must be greater than 0.
    key : JSONPathStr
        JSONPath expression used to extract the raw latency value (in milliseconds)
        from the trace. Defaults to the last interaction's metadata field
        ``latency_ms``.

    Examples
    --------
    >>> from giskard.checks import Latency, Scenario
    >>> scenario = (
    ...     Scenario(name="performance_test")
    ...     .interact(inputs="Quick question", outputs="Quick answer", metadata={"latency_ms": 150})
    ...     .check(Latency(max_seconds=1.0))
    ... )
    """

    max_seconds: float = Field(
        ...,
        gt=0,
        description="Maximum allowed response latency in seconds.",
    )
    key: JSONPathStr = Field(
        default="trace.last.metadata.latency_ms",
        description=(
            "JSONPath to extract the raw latency value (in milliseconds) from the trace. "
            "Defaults to the last interaction's metadata field 'latency_ms'."
        ),
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the latency check against the provided trace.

        Parameters
        ----------
        trace : TraceType
            The trace containing interaction history. The latency value is
            resolved from the path specified by ``key``.

        Returns
        -------
        CheckResult
            A passing result when the extracted latency is within ``max_seconds``,
            a failing result when it exceeds the threshold, and an error result
            when the key is missing from the trace.
        """
        value = resolve(trace, self.key)

        if isinstance(value, NoMatch):
            return CheckResult.error(
                message=(
                    f"No latency value found at '{self.key}'. "
                    "Ensure the interaction metadata contains the latency in milliseconds "
                    f"at the path '{self.key}'."
                ),
                details={"key": self.key},
            )

        try:
            latency_ms = float(value)
        except (TypeError, ValueError):
            return CheckResult.error(
                message=(
                    f"Latency value at '{self.key}' could not be converted to a number: "
                    f"{value!r} (type: {type(value).__name__})."
                ),
                details={"key": self.key, "raw_value": value},
            )

        actual_seconds = latency_ms / 1000.0
        metrics = [Metric(name="latency_seconds", value=actual_seconds)]
        details = {
            "actual_seconds": actual_seconds,
            "max_seconds": self.max_seconds,
            "latency_ms": latency_ms,
        }

        if actual_seconds <= self.max_seconds:
            return CheckResult(
                status=CheckStatus.PASS,
                message=(
                    f"Latency {actual_seconds:.3f}s is within the allowed "
                    f"threshold of {self.max_seconds}s."
                ),
                metrics=metrics,
                details=details,
            )

        return CheckResult(
            status=CheckStatus.FAIL,
            message=(
                f"Latency {actual_seconds:.3f}s exceeds the allowed "
                f"threshold of {self.max_seconds}s."
            ),
            metrics=metrics,
            details=details,
        )
