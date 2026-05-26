"""Unit tests for the Latency check."""

import pytest
from giskard.checks import CheckStatus, Interaction, Latency, Trace
from giskard.checks.core.result import Metric  # noqa: TC001


@pytest.mark.parametrize(
    "latency_ms,max_seconds",
    [(150, 1.0), (1000, 1.0), (499.9, 0.5)],
)
async def test_latency_pass(latency_ms, max_seconds):
    """Latency at or below threshold should pass."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": latency_ms})
    )
    result = await Latency(max_seconds=max_seconds).run(trace)
    assert result.status == CheckStatus.PASS


async def test_latency_fail_message():
    """Failure result should include actual and max latency."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 1500})
    )
    result = await Latency(max_seconds=1.0).run(trace)
    assert result.status == CheckStatus.FAIL
    assert "1.500s" in result.message
    assert "1.0s" in result.message


async def test_latency_metric_and_details():
    """Passing result should carry a latency_seconds metric and matching details."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 300})
    )
    result = await Latency(max_seconds=1.0).run(trace)
    assert result.status == CheckStatus.PASS
    assert len(result.metrics) == 1
    assert isinstance(result.metrics[0], Metric)
    assert result.metrics[0].name == "latency_seconds"
    assert pytest.approx(result.metrics[0].value) == 0.3
    assert pytest.approx(result.details["actual_seconds"]) == 0.3
    assert result.details["latency_ms"] == 300
    assert result.details["max_seconds"] == 1.0


async def test_latency_missing_key_errors():
    """Missing metadata key should produce an error with the key name in the message."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={})
    )
    result = await Latency(max_seconds=1.0).run(trace)
    assert result.status == CheckStatus.ERROR
    assert "latency_ms" in result.message


@pytest.mark.parametrize("bad_value", ["fast", None])
async def test_latency_non_numeric_errors(bad_value):
    """Non-numeric latency values should produce an error result."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": bad_value})
    )
    result = await Latency(max_seconds=1.0).run(trace)
    assert result.status == CheckStatus.ERROR


async def test_latency_custom_key():
    """Custom JSONPath key should be used to resolve the latency value."""
    trace = await Trace.from_interactions(
        Interaction(inputs="hello", outputs="hi", metadata={"response_time_ms": 200})
    )
    result = await Latency(
        max_seconds=1.0,
        key="trace.last.metadata.response_time_ms",
    ).run(trace)
    assert result.status == CheckStatus.PASS
