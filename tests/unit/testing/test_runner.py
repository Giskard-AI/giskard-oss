"""Tests for the test runner and result formatting utilities."""

from typing import Any

import pytest

from giskard.checks.core.check import Check, CheckResult, CheckStatus
from giskard.checks.core.context import Context
from giskard.checks.core.interaction_result import InteractionResult
from giskard.checks.generators import Interaction
from giskard.checks.generators.base import InteractionGenerator
from giskard.checks.testing.runner import TestCaseResult, TestRunner
from giskard.checks.testing.testcase import TestCase


class TestTestCaseResult:
    """Test the TestCaseResult class and its utility methods."""

    def test_format_failures_with_failed_checks(self):
        """Test format_failures with failed check results."""
        results = [
            CheckResult(
                status=CheckStatus.PASS,
                message="Check passed",
                details={"check_name": "test_check_1", "check_kind": "test_kind_1"},
            ),
            CheckResult(
                status=CheckStatus.FAIL,
                message="This check failed",
                details={"check_name": "test_check_2", "check_kind": "test_kind_2"},
            ),
            CheckResult(
                status=CheckStatus.ERROR,
                message="This check errored",
                details={"check_name": "test_check_3", "check_kind": "test_kind_3"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)
        failure_messages = test_result.format_failures()

        assert len(failure_messages) == 2
        assert "test_check_2 FAILED: This check failed" in failure_messages
        assert "test_check_3 ERRORED: This check errored" in failure_messages

    def test_format_failures_with_no_failures(self):
        """Test format_failures when all checks pass."""
        results = [
            CheckResult(
                status=CheckStatus.PASS,
                message="Check passed",
                details={"check_name": "test_check_1", "check_kind": "test_kind_1"},
            ),
            CheckResult(
                status=CheckStatus.SKIP,
                message="Check skipped",
                details={"check_name": "test_check_2", "check_kind": "test_kind_2"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)
        failure_messages = test_result.format_failures()

        assert len(failure_messages) == 0

    def test_format_failures_with_missing_check_info(self):
        """Test format_failures when check details are missing."""
        results = [
            CheckResult(
                status=CheckStatus.FAIL,
                message="Check failed without details",
                details={},
            ),
            CheckResult(
                status=CheckStatus.ERROR,
                message=None,
                details={"check_kind": "test_kind"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)
        failure_messages = test_result.format_failures()

        assert len(failure_messages) == 2
        assert "Unknown check FAILED: Check failed without details" in failure_messages
        assert (
            "test_kind ERRORED: No specific error message provided" in failure_messages
        )

    def test_format_failures_with_check_kind_fallback(self):
        """Test format_failures uses check_kind when check_name is not available."""
        results = [
            CheckResult(
                status=CheckStatus.FAIL,
                message="Check failed",
                details={"check_kind": "test_kind_only"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)
        failure_messages = test_result.format_failures()

        assert len(failure_messages) == 1
        assert "test_kind_only FAILED: Check failed" in failure_messages

    def test_assert_passed_with_success(self):
        """Test assert_passed does not raise when all checks pass."""
        results = [
            CheckResult(
                status=CheckStatus.PASS,
                message="Check passed",
                details={"check_name": "test_check_1", "check_kind": "test_kind_1"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)
        # Should not raise an exception
        test_result.assert_passed()

    def test_assert_passed_with_failures(self):
        """Test assert_passed raises AssertionError with formatted messages when checks fail."""
        results = [
            CheckResult(
                status=CheckStatus.PASS,
                message="Check passed",
                details={"check_name": "test_check_1", "check_kind": "test_kind_1"},
            ),
            CheckResult(
                status=CheckStatus.FAIL,
                message="This check failed",
                details={"check_name": "test_check_2", "check_kind": "test_kind_2"},
            ),
            CheckResult(
                status=CheckStatus.ERROR,
                message="This check errored",
                details={"check_name": "test_check_3", "check_kind": "test_kind_3"},
            ),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)

        with pytest.raises(AssertionError) as exc_info:
            test_result.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed with the following errors:" in error_message
        assert "test_check_2 FAILED: This check failed" in error_message
        assert "test_check_3 ERRORED: This check errored" in error_message

    def test_assert_passed_with_no_error_messages(self):
        """Test assert_passed handles case where no specific error messages are available."""
        results = [
            CheckResult(status=CheckStatus.FAIL, message=None, details={}),
        ]

        test_result = TestCaseResult(all_runs=[results], duration_ms=100)

        with pytest.raises(AssertionError) as exc_info:
            test_result.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed with the following errors:" in error_message
        assert (
            "Unknown check FAILED: No specific error message provided" in error_message
        )


class TestTestCase:
    """Test the TestCase class and its utility methods."""

    @pytest.mark.asyncio
    async def test_assert_passed_with_success(self):
        """Test assert_passed does not raise when all checks pass."""
        # Create a simple interaction and check that will pass
        interaction = Interaction(
            inputs={"input": "test"},
            outputs={"output": "test"},
        )

        # Create a mock check that always passes
        @Check.register("mock_passing")
        class MockPassingCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="Check passed",
                    details={"check_name": "mock_check", "check_kind": "mock_passing"},
                )

        test_case = TestCase(
            name="test_assert_passed_success",
            interaction=interaction,
            checks=[MockPassingCheck()],
        )

        # Should not raise an exception
        await test_case.assert_passed()

    @pytest.mark.asyncio
    async def test_assert_passed_with_failures(self):
        """Test assert_passed raises AssertionError with formatted messages when checks fail."""
        # Create a simple interaction
        interaction = Interaction(
            inputs={"input": "test"},
            outputs={"output": "test"},
        )

        # Create a mock check that always fails
        @Check.register("mock_failing")
        class MockFailingCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.FAIL,
                    message="This check failed",
                    details={"check_name": "mock_check", "check_kind": "mock_failing"},
                )

        test_case = TestCase(
            name="test_assert_passed_failure",
            interaction=interaction,
            checks=[MockFailingCheck()],
        )

        with pytest.raises(AssertionError) as exc_info:
            await test_case.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed with the following errors:" in error_message
        assert "mock_failing FAILED: This check failed" in error_message

    @pytest.mark.asyncio
    async def test_assert_passed_with_errors(self):
        """Test assert_passed raises AssertionError when checks error."""
        # Create a simple interaction
        interaction = Interaction(
            inputs={"input": "test"},
            outputs={"output": "test"},
        )

        # Create a mock check that always errors
        @Check.register("mock_erroring")
        class MockErroringCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.ERROR,
                    message="This check errored",
                    details={"check_name": "mock_check", "check_kind": "mock_erroring"},
                )

        test_case = TestCase(
            name="test_assert_passed_error",
            interaction=interaction,
            checks=[MockErroringCheck()],
        )

        with pytest.raises(AssertionError) as exc_info:
            await test_case.assert_passed()

        error_message = str(exc_info.value)
        assert "Test case failed with the following errors:" in error_message
        assert "mock_erroring ERRORED: This check errored" in error_message


class CountingInteractionGenerator(InteractionGenerator):
    """Helper class to track how many times generate() is called."""

    call_count: int = 0
    inputs: str = "test_input"
    outputs: str = "test_output"

    async def generate(self, context: Context) -> InteractionResult[str, str]:
        """Generate an interaction and track the call count."""
        self.call_count += 1
        return InteractionResult(
            inputs=self.inputs,
            outputs=self.outputs,
            metadata={"call_count": self.call_count},
        )


class TestTestRunnerMaxRuns:
    """Test the TestRunner behavior with max_runs > 1."""

    @pytest.mark.asyncio
    async def test_early_stop_on_first_failure(self):
        """Test that runner stops after first failure and doesn't call generator again."""
        # Create a counting generator to track calls
        counting_generator = CountingInteractionGenerator()

        # Create a check that always fails
        @Check.register("mock_failing_max_runs")
        class MockFailingCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.FAIL,
                    message="This check always fails",
                    details={
                        "check_name": "failing_check",
                        "check_kind": "mock_failing_max_runs",
                    },
                )

        test_case = TestCase(
            name="test_early_stop_failure",
            interaction=counting_generator,
            checks=[MockFailingCheck()],
        )

        # Run with max_runs=3, should stop after first failure
        result = await test_case.run(max_runs=3)

        # Assertions
        assert result.total_runs == 1, "Should stop after first failure"
        assert counting_generator.call_count == 1, (
            "Generator should be called only once"
        )
        assert not result.passed, "Result should not be passed"
        assert result.failed, "Result should be failed"
        assert not result.errored, "Result should not be errored"

    @pytest.mark.asyncio
    async def test_all_runs_complete_on_success(self):
        """Test that runner completes all runs when checks pass."""
        # Create a counting generator to track calls
        counting_generator = CountingInteractionGenerator()

        # Create a check that always passes
        @Check.register("mock_passing_max_runs")
        class MockPassingCheck(Check):
            async def run(
                self, interaction: InteractionResult[Any, Any]
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="This check always passes",
                    details={
                        "check_name": "passing_check",
                        "check_kind": "mock_passing_max_runs",
                    },
                )

        test_case = TestCase(
            name="test_all_runs_success",
            interaction=counting_generator,
            checks=[MockPassingCheck()],
        )

        # Run with max_runs=5, should complete all runs
        result = await test_case.run(max_runs=5)

        # Assertions
        assert result.total_runs == 5, "Should complete all 5 runs"
        assert counting_generator.call_count == 5, "Generator should be called 5 times"
        assert result.passed, "Result should be passed"
        assert not result.failed, "Result should not be failed"
        assert not result.errored, "Result should not be errored"

    @pytest.mark.asyncio
    async def test_early_stop_on_error(self):
        """Test that runner stops after first error and doesn't call generator again."""
        # Create a counting generator to track calls
        counting_generator = CountingInteractionGenerator()

        # Create a check that always errors
        @Check.register("mock_erroring_max_runs")
        class MockErroringCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.ERROR,
                    message="This check always errors",
                    details={
                        "check_name": "erroring_check",
                        "check_kind": "mock_erroring_max_runs",
                    },
                )

        test_case = TestCase(
            name="test_early_stop_error",
            interaction=counting_generator,
            checks=[MockErroringCheck()],
        )

        # Run with max_runs=3, should stop after first error
        result = await test_case.run(max_runs=3)

        # Assertions
        assert result.total_runs == 1, "Should stop after first error"
        assert counting_generator.call_count == 1, (
            "Generator should be called only once"
        )
        assert not result.passed, "Result should not be passed"
        assert not result.failed, "Result should not be failed"
        assert result.errored, "Result should be errored"

    @pytest.mark.asyncio
    async def test_mixed_checks_with_early_stop(self):
        """Test that runner stops after first failure even with multiple checks."""
        # Create a counting generator to track calls
        counting_generator = CountingInteractionGenerator()

        # Create multiple checks: pass, fail, pass
        @Check.register("mock_passing_mixed")
        class MockPassingCheck(Check):
            async def run(
                self, interaction: InteractionResult[Any, Any]
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="This check passes",
                    details={
                        "check_name": "passing_check",
                        "check_kind": "mock_passing_mixed",
                    },
                )

        @Check.register("mock_failing_mixed")
        class MockFailingCheck(Check):
            async def run(
                self,
                interaction: InteractionResult[Any, Any],
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.FAIL,
                    message="This check fails",
                    details={
                        "check_name": "failing_check",
                        "check_kind": "mock_failing_mixed",
                    },
                )

        @Check.register("mock_passing_mixed_2")
        class MockPassingCheck2(Check):
            async def run(
                self, interaction: InteractionResult[Any, Any]
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="This check also passes",
                    details={
                        "check_name": "passing_check_2",
                        "check_kind": "mock_passing_mixed_2",
                    },
                )

        test_case = TestCase(
            name="test_mixed_checks_early_stop",
            interaction=counting_generator,
            checks=[MockPassingCheck(), MockFailingCheck(), MockPassingCheck2()],
        )

        # Run with max_runs=4, should stop after first run due to failing check
        result = await test_case.run(max_runs=4)

        # Assertions
        assert result.total_runs == 1, "Should stop after first run with failure"
        assert counting_generator.call_count == 1, (
            "Generator should be called only once"
        )
        assert not result.passed, "Result should not be passed"
        assert result.failed, "Result should be failed"
        assert not result.errored, "Result should not be errored"

        # Verify all 3 checks were executed in the single run
        assert len(result.results) == 3, "All 3 checks should be executed"
        assert result.results[0].passed, "First check should pass"
        assert result.results[1].failed, "Second check should fail"
        assert result.results[2].passed, "Third check should pass"

    @pytest.mark.asyncio
    async def test_runner_directly_with_max_runs(self):
        """Test TestRunner.run() directly with max_runs > 1."""
        # Create a counting generator to track calls
        counting_generator = CountingInteractionGenerator()

        # Create a check that always passes
        @Check.register("mock_passing_direct")
        class MockPassingCheck(Check):
            async def run(
                self, interaction: InteractionResult[Any, Any]
            ) -> CheckResult:
                _ = interaction  # Mark as used to avoid linting warning
                return CheckResult(
                    status=CheckStatus.PASS,
                    message="This check always passes",
                    details={
                        "check_name": "passing_check",
                        "check_kind": "mock_passing_direct",
                    },
                )

        test_case = TestCase(
            name="test_runner_direct",
            interaction=counting_generator,
            checks=[MockPassingCheck()],
        )

        # Use TestRunner directly
        runner = TestRunner()
        result = await runner.run(test_case, max_runs=3)

        # Assertions
        assert result.total_runs == 3, "Should complete all 3 runs"
        assert counting_generator.call_count == 3, "Generator should be called 3 times"
        assert result.passed, "Result should be passed"
        assert not result.failed, "Result should not be failed"
        assert not result.errored, "Result should not be errored"
