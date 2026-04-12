"""Unit tests for the Latency check.

Tests cover:
- Pass when latency is below threshold
- Pass when latency equals threshold (boundary)
- Fail when latency exceeds threshold
- Error when metadata key is missing
- Error when value cannot be converted to float
- Actual latency reported as a Metric
- Custom key path
"""

import pytest
from giskard.checks import CheckStatus, Interaction, Latency, Trace
from giskard.checks.core.result import Metric  # noqa: TC001


class TestLatencyPass:
    """Latency check passes when latency is within the threshold."""

    async def test_below_threshold_passes(self):
        """Latency well below threshold should pass."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 150})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_at_threshold_passes(self):
        """Latency exactly at threshold should pass (boundary: <=)."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 1000})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_float_ms_below_threshold_passes(self):
        """Fractional millisecond value below threshold should pass."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 499.9})
        )
        check = Latency(max_seconds=0.5)

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed


class TestLatencyFail:
    """Latency check fails when latency exceeds the threshold."""

    async def test_above_threshold_fails(self):
        """Latency above threshold should fail."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 2000})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed

    async def test_failure_message_contains_latency(self):
        """Failure message should mention actual and max latency."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 1500})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.failed
        assert result.message is not None
        assert "1.500s" in result.message
        assert "1.0s" in result.message

    async def test_just_above_threshold_fails(self):
        """Latency just 1 ms above threshold should fail."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 1001})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed


class TestLatencyMetric:
    """Actual latency is reported as a Metric."""

    async def test_metric_reported_on_pass(self):
        """A latency_seconds metric is attached to passing results."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 300})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.passed
        assert len(result.metrics) == 1
        metric = result.metrics[0]
        assert isinstance(metric, Metric)
        assert metric.name == "latency_seconds"
        assert pytest.approx(metric.value) == 0.3

    async def test_metric_reported_on_fail(self):
        """A latency_seconds metric is attached to failing results."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 2500})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.failed
        assert len(result.metrics) == 1
        metric = result.metrics[0]
        assert metric.name == "latency_seconds"
        assert pytest.approx(metric.value) == 2.5

    async def test_details_contain_latency_info(self):
        """Result details should contain actual_seconds, max_seconds, and latency_ms."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"latency_ms": 400})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.details["latency_ms"] == 400
        assert pytest.approx(result.details["actual_seconds"]) == 0.4
        assert result.details["max_seconds"] == 1.0


class TestLatencyMissingKey:
    """Latency check errors when the metadata key is missing."""

    async def test_missing_latency_key_errors(self):
        """No latency_ms in metadata should produce an error result."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert result.errored

    async def test_missing_key_error_message(self):
        """Error message should indicate where the latency was expected."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.errored
        assert result.message is not None
        assert "latency_ms" in result.message

    async def test_wrong_nested_key_errors(self):
        """Key pointing to a non-existent nested path should error."""
        trace = await Trace.from_interactions(
            Interaction(inputs="hello", outputs="hi", metadata={"timing": 500})
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.errored


class TestLatencyCustomKey:
    """Latency check supports a custom JSONPath key."""

    async def test_custom_key_passes(self):
        """Custom key pointing to a valid latency value should pass."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="hello",
                outputs="hi",
                metadata={"response_time_ms": 200},
            )
        )
        check = Latency(
            max_seconds=1.0,
            key="trace.interactions[-1].metadata.response_time_ms",
        )

        result = await check.run(trace)

        assert result.passed

    async def test_custom_key_fails(self):
        """Custom key pointing to a value above threshold should fail."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="hello",
                outputs="hi",
                metadata={"response_time_ms": 3000},
            )
        )
        check = Latency(
            max_seconds=1.0,
            key="trace.interactions[-1].metadata.response_time_ms",
        )

        result = await check.run(trace)

        assert result.failed


class TestLatencyNonNumericValue:
    """Latency check errors when the value cannot be converted to float."""

    async def test_string_value_errors(self):
        """A string latency value that is not numeric should error."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="hello",
                outputs="hi",
                metadata={"latency_ms": "fast"},
            )
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.errored

    async def test_none_value_errors(self):
        """A None latency value should error."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs="hello",
                outputs="hi",
                metadata={"latency_ms": None},
            )
        )
        check = Latency(max_seconds=1.0)

        result = await check.run(trace)

        assert result.errored
