"""Tests for the test runner and result formatting utilities."""

from typing import Any

import pytest

from giskard_checks.core.check import Check, CheckResult, CheckStatus
from giskard_checks.core.interactions import Interaction
from giskard_checks.testing.runner import TestCaseResult
from giskard_checks.testing.testcase import TestCase


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

        test_result = TestCaseResult(results=results, duration_ms=100)
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

        test_result = TestCaseResult(results=results, duration_ms=100)
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

        test_result = TestCaseResult(results=results, duration_ms=100)
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

        test_result = TestCaseResult(results=results, duration_ms=100)
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

        test_result = TestCaseResult(results=results, duration_ms=100)
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

        test_result = TestCaseResult(results=results, duration_ms=100)

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

        test_result = TestCaseResult(results=results, duration_ms=100)

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
            async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
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
            async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
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
            async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
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
