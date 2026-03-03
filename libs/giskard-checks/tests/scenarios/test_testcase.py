"""Unit tests for TestCase class.

Tests cover normal cases, edge cases, error handling, and max_runs functionality
for test case execution.
"""

from typing import override

import pytest
from giskard.checks import (
    Check,
    CheckResult,
    Equals,
    Interact,
    Interaction,
    TestCase,
    Trace,
)

# Test Classes


class TestTestCaseNormalCases:
    """Test normal execution paths for test cases."""

    async def test_testcase_with_single_passing_check(self):
        """Test test case with a single check that passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="test_input", outputs="test_output")
        )
        check = Equals(
            expected_value="test_output",
            key="trace.interactions[-1].outputs",
        )
        test_case = TestCase(
            name="single_check",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert len(result.results) == 1
        assert result.results[0].passed
        assert result.passed
        assert not result.failed
        assert not result.errored
        assert result.duration_ms >= 0

    async def test_testcase_with_multiple_passing_checks(self):
        """Test test case with multiple checks that all pass."""
        trace = await Trace.from_interactions(
            Interaction(inputs={"value": 42}, outputs={"result": 42, "status": "ok"})
        )
        checks = [
            Equals(
                expected_value=42,
                key="trace.interactions[-1].outputs.result",
            ),
            Equals(
                expected_value="ok",
                key="trace.interactions[-1].outputs.status",
            ),
        ]
        test_case = TestCase(
            name="multiple_checks",
            trace=trace,
            checks=checks,
        )

        result = await test_case.run()

        assert len(result.results) == 2
        assert all(r.passed for r in result.results)
        assert result.passed

    async def test_testcase_with_failing_check(self):
        """Test test case with a failing check."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="expected", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="failing_check",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert len(result.results) == 1
        assert result.results[0].failed
        assert result.failed
        assert not result.passed
        assert not result.errored

    async def test_testcase_with_multiple_checks_one_fails(self):
        """Test test case with multiple checks where one fails - all checks should run."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        checks = [
            Equals(
                expected_value="output",
                key="trace.interactions[-1].outputs",
            ),  # Passes
            Equals(
                expected_value="wrong",
                key="trace.interactions[-1].outputs",
            ),  # Fails
            Equals(
                expected_value="output",
                key="trace.interactions[-1].outputs",
            ),  # Passes
        ]
        test_case = TestCase(
            name="mixed_results",
            trace=trace,
            checks=checks,
        )

        result = await test_case.run()

        assert len(result.results) == 3  # All checks run even if one fails
        assert result.results[0].passed
        assert result.results[1].failed
        assert result.results[2].passed
        assert result.failed

    async def test_testcase_with_multiple_failures_all_run(self):
        """Test that all checks run even when multiple checks fail."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        checks = [
            Equals(
                expected_value="output",
                key="trace.interactions[-1].outputs",
            ),  # Passes
            Equals(
                expected_value="wrong1",
                key="trace.interactions[-1].outputs",
            ),  # Fails
            Equals(
                expected_value="wrong2",
                key="trace.interactions[-1].outputs",
            ),  # Fails
            Equals(
                expected_value="output",
                key="trace.interactions[-1].outputs",
            ),  # Passes
        ]
        test_case = TestCase(
            name="multiple_failures",
            trace=trace,
            checks=checks,
        )

        result = await test_case.run()

        assert len(result.results) == 4  # All checks run
        assert result.results[0].passed
        assert result.results[1].failed
        assert result.results[2].failed
        assert result.results[3].passed
        assert result.failed

    async def test_testcase_without_name(self):
        """Test test case without a name."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="output", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name=None,
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert result.passed
        assert len(result.results) == 1


class TestTestCaseResult:
    """Test TestCaseResult properties and methods."""

    async def test_testcase_result_properties(self):
        """Test TestCaseResult convenience properties."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="output", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="properties_test",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert result.passed
        assert not result.failed
        assert not result.errored
        assert not result.skipped
        assert len(result.results) == 1

    async def test_testcase_result_format_failures(self):
        """Test format_failures() method."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        checks = [
            Equals(
                expected_value="output",
                key="trace.interactions[-1].outputs",
            ),  # Passes
            Equals(
                expected_value="wrong",
                key="trace.interactions[-1].outputs",
            ),  # Fails
        ]
        test_case = TestCase(
            name="format_failures_test",
            trace=trace,
            checks=checks,
        )

        result = await test_case.run()

        failures = result.format_failures()
        assert len(failures) == 1
        assert "FAILED" in failures[0]
        assert "wrong" in failures[0] or "output" in failures[0]

    async def test_testcase_result_format_failures_with_errors(self):
        """Test format_failures() with error results."""

        @Check.register("erroring_check_format_test")
        class ErroringCheckFormat(Check[str, str, Trace[str, str]]):
            @override
            async def run(self, trace: Trace[str, str]) -> CheckResult:
                raise ValueError("Test error")

        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = ErroringCheckFormat()
        test_case = TestCase(
            name="format_errors_test",
            trace=trace,
            checks=[check],
        )

        with pytest.raises(ValueError, match="Test error"):
            _ = await test_case.run()

        # Test with return_exception=True to return the exception
        result = await test_case.run(return_exception=True)

        failures = result.format_failures()
        assert len(failures) == 1
        assert "ERRORED" in failures[0]
        assert "error" in failures[0].lower()

    async def test_testcase_result_format_failures_no_failures(self):
        """Test format_failures() when all checks pass."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="output", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="no_failures_test",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        failures = result.format_failures()
        assert len(failures) == 0

    async def test_testcase_result_assert_passed_success(self):
        """Test assert_passed() when test case passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="output", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="assert_passed_success",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        # Should not raise
        result.assert_passed()

    async def test_testcase_result_assert_passed_failure(self):
        """Test assert_passed() raises AssertionError when test case fails."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="wrong", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="assert_passed_failure",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        with pytest.raises(AssertionError) as exc_info:
            result.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed" in error_message
        assert "FAILED" in error_message


class TestTestCaseAssertPassed:
    """Test TestCase.assert_passed() convenience method."""

    async def test_testcase_assert_passed_success(self):
        """Test assert_passed() when test case passes."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="output", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="assert_passed_success",
            trace=trace,
            checks=[check],
        )

        # Should not raise
        await test_case.assert_passed()

    async def test_testcase_assert_passed_failure(self):
        """Test assert_passed() raises AssertionError when test case fails."""
        trace = await Trace.from_interactions(
            Interaction(inputs="input", outputs="output")
        )
        check = Equals(expected_value="wrong", key="trace.interactions[-1].outputs")
        test_case = TestCase(
            name="assert_passed_failure",
            trace=trace,
            checks=[check],
        )

        with pytest.raises(AssertionError) as exc_info:
            await test_case.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed" in error_message


class TestTestCaseEdgeCases:
    """Test edge cases for test cases."""

    async def test_testcase_with_complex_interaction(self):
        """Test test case with complex interaction data."""
        trace = await Trace.from_interactions(
            Interaction(
                inputs={"user": "Alice", "message": "Hello"},
                outputs={"assistant": "Bob", "response": "Hi Alice!"},
                metadata={"timestamp": "2024-01-01"},
            )
        )
        check = Equals(
            expected_value="Hi Alice!",
            key="trace.interactions[-1].outputs.response",
        )
        test_case = TestCase(
            name="complex_interaction",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert result.passed
        assert len(result.results) == 1

    async def test_testcase_with_dynamic_outputs(self):
        """Test test case with callable outputs."""

        def output_func(inputs: str) -> str:
            return f"Processed: {inputs}"

        trace = await Trace.from_interactions(
            Interact(inputs="test", outputs=output_func)
        )
        check = Equals(
            expected_value="Processed: test",
            key="trace.interactions[-1].outputs",
        )
        test_case = TestCase(
            name="dynamic_outputs",
            trace=trace,
            checks=[check],
        )

        result = await test_case.run()

        assert result.passed

    async def test_testcase_no_checks(self):
        """Test test case with no checks."""
        trace = Trace(interactions=[Interaction(inputs="input", outputs="output")])

        test_case = TestCase(
            name="no_checks",
            trace=trace,
            checks=[],
        )

        result = await test_case.run()

        assert len(result.results) == 0
        assert result.passed  # No checks means passed
        assert not result.failed
        assert not result.errored
        assert not result.skipped
