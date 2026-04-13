"""Unit tests for AllOf, AnyOf, and Not composition checks.

Tests cover:
- AllOf: all pass, partial pass (short-circuit), all fail
- AnyOf: first passes, last passes, all fail
- Not: invert pass, invert fail, pass-through error/skip
- Nested composition (AllOf inside AnyOf, Not wrapping AllOf)
- Serialisation round-trip via model_dump / model_validate
"""

from typing import Any

from giskard.checks import (
    AllOf,
    AnyOf,
    Equals,
    Interaction,
    LesserThan,
    Not,
    Trace,
)
from giskard.checks.builtin.fn import FnCheck
from giskard.checks.core.result import CheckResult
from giskard.checks.core.result import CheckStatus as CS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_fn_check(message: str = "passed") -> FnCheck[Any, Any, Trace[Any, Any]]:
    async def _fn(trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.success(message=message)

    return FnCheck(fn=_fn)


def _failing_fn_check(message: str = "failed") -> FnCheck[Any, Any, Trace[Any, Any]]:
    async def _fn(trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.failure(message=message)

    return FnCheck(fn=_fn)


def _error_fn_check(message: str = "error") -> FnCheck[Any, Any, Trace[Any, Any]]:
    async def _fn(trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.error(message=message)

    return FnCheck(fn=_fn)


def _skip_fn_check(message: str = "skip") -> FnCheck[Any, Any, Trace[Any, Any]]:
    async def _fn(trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.skip(message=message)

    return FnCheck(fn=_fn)


# ---------------------------------------------------------------------------
# AllOf
# ---------------------------------------------------------------------------


class TestAllOf:
    """AllOf passes only when every inner check passes."""

    async def test_all_pass(self):
        """All checks pass → combined success."""
        check = AllOf(checks=[_passing_fn_check("a"), _passing_fn_check("b")])
        result = await check.run(Trace())

        assert result.status == CS.PASS
        assert result.passed

    async def test_first_fails_short_circuits(self):
        """First failure stops evaluation; subsequent checks are not run."""
        call_log: list[str] = []

        async def _fail(trace: Trace[Any, Any]) -> CheckResult:
            call_log.append("fail")
            return CheckResult.failure(message="first failed")

        async def _should_not_run(trace: Trace[Any, Any]) -> CheckResult:
            call_log.append("second")
            return CheckResult.success(message="second")

        check = AllOf(checks=[FnCheck(fn=_fail), FnCheck(fn=_should_not_run)])
        result = await check.run(Trace())

        assert result.status == CS.FAIL
        assert result.failed
        assert "first failed" in (result.message or "")
        assert "second" not in call_log

    async def test_last_fails(self):
        """Failure in last check is returned."""
        check = AllOf(checks=[_passing_fn_check(), _failing_fn_check("last failed")])
        result = await check.run(Trace())

        assert result.failed
        assert "last failed" in (result.message or "")

    async def test_all_fail(self):
        """First failure is returned immediately."""
        check = AllOf(checks=[_failing_fn_check("first"), _failing_fn_check("second")])
        result = await check.run(Trace())

        assert result.failed
        assert "first" in (result.message or "")

    async def test_error_short_circuits(self):
        """An erroring check stops evaluation."""
        check = AllOf(checks=[_error_fn_check("boom"), _passing_fn_check()])
        result = await check.run(Trace())

        assert result.status == CS.ERROR
        assert result.errored

    async def test_single_check_pass(self):
        """Single passing check returns success."""
        check = AllOf(checks=[_passing_fn_check("only")])
        result = await check.run(Trace())

        assert result.passed

    async def test_combined_message_contains_all_messages(self):
        """Success message contains messages from all inner checks."""
        check = AllOf(checks=[_passing_fn_check("msg_a"), _passing_fn_check("msg_b")])
        result = await check.run(Trace())

        assert result.passed
        assert "msg_a" in (result.message or "")
        assert "msg_b" in (result.message or "")

    async def test_with_real_checks(self):
        """AllOf works with real built-in checks."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AllOf(
            checks=[
                LesserThan(expected_value=10, key="trace.last.outputs"),
                Equals(expected_value=5, key="trace.last.outputs"),
            ]
        )
        result = await check.run(trace)

        assert result.passed

    async def test_with_real_checks_partial_fail(self):
        """AllOf fails when one real check fails."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AllOf(
            checks=[
                LesserThan(expected_value=10, key="trace.last.outputs"),
                Equals(expected_value=99, key="trace.last.outputs"),  # will fail
            ]
        )
        result = await check.run(trace)

        assert result.failed


# ---------------------------------------------------------------------------
# AnyOf
# ---------------------------------------------------------------------------


class TestAnyOf:
    """AnyOf passes when at least one inner check passes."""

    async def test_first_passes_short_circuits(self):
        """First success stops evaluation."""
        call_log: list[str] = []

        async def _pass(trace: Trace[Any, Any]) -> CheckResult:
            call_log.append("first")
            return CheckResult.success(message="first passed")

        async def _should_not_run(trace: Trace[Any, Any]) -> CheckResult:
            call_log.append("second")
            return CheckResult.success(message="second")

        check = AnyOf(checks=[FnCheck(fn=_pass), FnCheck(fn=_should_not_run)])
        result = await check.run(Trace())

        assert result.passed
        assert "second" not in call_log

    async def test_last_passes(self):
        """Last check passing returns success."""
        check = AnyOf(checks=[_failing_fn_check(), _passing_fn_check("last")])
        result = await check.run(Trace())

        assert result.passed

    async def test_all_fail(self):
        """All checks failing returns combined failure."""
        check = AnyOf(checks=[_failing_fn_check("f1"), _failing_fn_check("f2")])
        result = await check.run(Trace())

        assert result.failed
        assert "f1" in (result.message or "")
        assert "f2" in (result.message or "")

    async def test_single_check_pass(self):
        """Single passing check returns success."""
        check = AnyOf(checks=[_passing_fn_check()])
        result = await check.run(Trace())

        assert result.passed

    async def test_single_check_fail(self):
        """Single failing check returns failure."""
        check = AnyOf(checks=[_failing_fn_check("only")])
        result = await check.run(Trace())

        assert result.failed

    async def test_with_real_checks(self):
        """AnyOf works with real built-in checks."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AnyOf(
            checks=[
                Equals(expected_value=99, key="trace.last.outputs"),  # fail
                Equals(expected_value=5, key="trace.last.outputs"),  # pass
            ]
        )
        result = await check.run(trace)

        assert result.passed

    async def test_with_real_checks_all_fail(self):
        """AnyOf fails when no real check passes."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AnyOf(
            checks=[
                Equals(expected_value=99, key="trace.last.outputs"),
                Equals(expected_value=100, key="trace.last.outputs"),
            ]
        )
        result = await check.run(trace)

        assert result.failed


# ---------------------------------------------------------------------------
# Not
# ---------------------------------------------------------------------------


class TestNot:
    """Not inverts the pass/fail of the inner check."""

    async def test_inverts_pass_to_fail(self):
        """A passing inner check becomes a failure."""
        check = Not(check=_passing_fn_check("inner passed"))
        result = await check.run(Trace())

        assert result.failed

    async def test_inverts_fail_to_pass(self):
        """A failing inner check becomes a success."""
        check = Not(check=_failing_fn_check("inner failed"))
        result = await check.run(Trace())

        assert result.passed

    async def test_error_passed_through(self):
        """An error result is not inverted."""
        check = Not(check=_error_fn_check("boom"))
        result = await check.run(Trace())

        assert result.status == CS.ERROR
        assert result.errored

    async def test_skip_passed_through(self):
        """A skip result is not inverted."""
        check = Not(check=_skip_fn_check("skipped"))
        result = await check.run(Trace())

        assert result.status == CS.SKIP
        assert result.skipped

    async def test_failure_message_mentions_original(self):
        """Inverted-pass failure message references original message."""
        check = Not(check=_passing_fn_check("some message"))
        result = await check.run(Trace())

        assert result.failed
        assert "some message" in (result.message or "")

    async def test_success_message_mentions_original(self):
        """Inverted-fail success message references original message."""
        check = Not(check=_failing_fn_check("bad thing"))
        result = await check.run(Trace())

        assert result.passed
        assert "bad thing" in (result.message or "")

    async def test_with_real_check(self):
        """Not works with a real built-in check."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        # Equals(99) fails → Not inverts to pass
        check = Not(check=Equals(expected_value=99, key="trace.last.outputs"))
        result = await check.run(trace)

        assert result.passed


# ---------------------------------------------------------------------------
# Nested composition
# ---------------------------------------------------------------------------


class TestNestedComposition:
    """Composition operators can be nested arbitrarily."""

    async def test_all_of_inside_any_of(self):
        """AllOf nested inside AnyOf."""
        check = AnyOf(
            checks=[
                AllOf(checks=[_failing_fn_check(), _passing_fn_check()]),  # fails
                AllOf(
                    checks=[_passing_fn_check("x"), _passing_fn_check("y")]
                ),  # passes
            ]
        )
        result = await check.run(Trace())

        assert result.passed

    async def test_not_wrapping_all_of(self):
        """Not wrapping an AllOf that partially fails."""
        # AllOf fails (second check fails) → Not inverts to pass
        check = Not(check=AllOf(checks=[_passing_fn_check(), _failing_fn_check()]))
        result = await check.run(Trace())

        assert result.passed

    async def test_all_of_with_not(self):
        """AllOf containing a Not check."""
        # Not(_failing_fn_check) → pass; combined with another pass → AllOf pass
        check = AllOf(
            checks=[
                Not(check=_failing_fn_check()),
                _passing_fn_check(),
            ]
        )
        result = await check.run(Trace())

        assert result.passed


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    """Composition checks serialise and deserialise correctly."""

    async def test_all_of_serialises(self):
        """AllOf round-trips through model_dump."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=3))
        check = AllOf(checks=[LesserThan(expected_value=10, key="trace.last.outputs")])
        data = check.model_dump()

        assert data["kind"] == "all_of"
        assert data["checks"][0]["kind"] == "lesser_than"

        restored = AllOf.model_validate(data)
        result = await restored.run(trace)
        assert result.passed

    async def test_any_of_serialises(self):
        """AnyOf round-trips through model_dump."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AnyOf(
            checks=[
                Equals(expected_value=99, key="trace.last.outputs"),
                Equals(expected_value=5, key="trace.last.outputs"),
            ]
        )
        data = check.model_dump()

        assert data["kind"] == "any_of"

        restored = AnyOf.model_validate(data)
        result = await restored.run(trace)
        assert result.passed

    async def test_not_serialises(self):
        """Not round-trips through model_dump."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = Not(check=Equals(expected_value=99, key="trace.last.outputs"))
        data = check.model_dump()

        assert data["kind"] == "not"
        assert data["check"]["kind"] == "equals"

        restored = Not.model_validate(data)
        result = await restored.run(trace)
        assert result.passed

    async def test_nested_serialises(self):
        """Nested composition round-trips correctly."""
        trace = await Trace.from_interactions(Interaction(inputs="q", outputs=5))
        check = AllOf(
            checks=[
                Not(check=Equals(expected_value=99, key="trace.last.outputs")),
                LesserThan(expected_value=10, key="trace.last.outputs"),
            ]
        )
        data = check.model_dump()
        restored = AllOf.model_validate(data)
        result = await restored.run(trace)

        assert result.passed
