"""Tests for the test runner and result formatting utilities."""

import pytest

from giskard_checks.core.check import CheckResult, CheckStatus
from giskard_checks.testing.runner import TestCaseResult


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
