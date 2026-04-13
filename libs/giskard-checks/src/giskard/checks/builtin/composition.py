from typing import override

from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.result import CheckResult, CheckStatus


@Check.register("all_of")
class AllOf[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Passes only when **all** inner checks pass (short-circuits on first failure).

    Runs each check in order. The first failing or erroring check stops
    evaluation and its result is returned immediately. If all checks pass,
    a combined success result is returned.

    Attributes
    ----------
    checks : list[Check]
        Ordered list of checks to evaluate. All must pass for this check to pass.

    Examples
    --------
    >>> from giskard.checks import AllOf, LesserThan, Equals
    >>> check = AllOf(checks=[
    ...     LesserThan(expected_value=10, key="trace.last.outputs"),
    ...     Equals(expected_value=5, key="trace.last.outputs"),
    ... ])
    """

    checks: list[Check[InputType, OutputType, TraceType]] = Field(
        ...,
        description="Ordered list of checks to evaluate. All must pass.",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Run all checks in order, short-circuiting on the first failure.

        Parameters
        ----------
        trace : TraceType
            The trace to evaluate against.

        Returns
        -------
        CheckResult
            The first non-passing result, or a combined success if all pass.
        """
        passed_messages: list[str] = []

        for check in self.checks:
            result = await check.run(trace)
            if not result.passed:
                return result
            if result.message:
                passed_messages.append(result.message)

        return CheckResult.success(
            message="; ".join(passed_messages)
            if passed_messages
            else "All checks passed.",
        )


@Check.register("any_of")
class AnyOf[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Passes when **at least one** inner check passes (short-circuits on first pass).

    Runs each check in order. The first passing check stops evaluation and
    its result is returned immediately. If all checks fail, a combined failure
    result is returned.

    Attributes
    ----------
    checks : list[Check]
        Ordered list of checks to evaluate. At least one must pass.

    Examples
    --------
    >>> from giskard.checks import AnyOf, StringMatching
    >>> check = AnyOf(checks=[
    ...     StringMatching(keyword="yes", key="trace.last.outputs"),
    ...     StringMatching(keyword="approved", key="trace.last.outputs"),
    ... ])
    """

    checks: list[Check[InputType, OutputType, TraceType]] = Field(
        ...,
        description="Ordered list of checks to evaluate. At least one must pass.",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Run checks in order, short-circuiting on the first passing result.

        Parameters
        ----------
        trace : TraceType
            The trace to evaluate against.

        Returns
        -------
        CheckResult
            The first passing result, or a combined failure if none pass.
        """
        failure_messages: list[str] = []
        all_skipped = True

        for check in self.checks:
            result = await check.run(trace)
            if result.passed or result.errored:
                return result
            if not result.skipped:
                all_skipped = False
            if result.message:
                failure_messages.append(result.message)

        if all_skipped and self.checks:
            return CheckResult.skip(
                message="No checks passed and all were skipped."
                + (
                    f" Details: {'; '.join(failure_messages)}"
                    if failure_messages
                    else ""
                ),
            )

        return CheckResult.failure(
            message=(
                "No checks passed. Failures: " + "; ".join(failure_messages)
                if failure_messages
                else "All checks failed."
            ),
        )


@Check.register("not")
class Not[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Inverts the result of an inner check.

    A passing inner result becomes a failure, and a failing inner result
    becomes a pass. Error and skip results are passed through unchanged.

    Attributes
    ----------
    check : Check
        The inner check whose result will be inverted.

    Examples
    --------
    >>> from giskard.checks import Not, StringMatching
    >>> check = Not(check=StringMatching(keyword="forbidden", key="trace.last.outputs"))
    """

    check: Check[InputType, OutputType, TraceType] = Field(
        ...,
        description="The inner check whose result will be inverted.",
    )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Run the inner check and invert its pass/fail result.

        Error and skip results are passed through without inversion.

        Parameters
        ----------
        trace : TraceType
            The trace to evaluate against.

        Returns
        -------
        CheckResult
            Inverted result (pass→fail, fail→pass). Error/skip unchanged.
        """
        result = await self.check.run(trace)

        if result.status in (CheckStatus.ERROR, CheckStatus.SKIP):
            return result

        if result.passed:
            return CheckResult.failure(
                message=(
                    "Expected check to fail, but it passed"
                    + (f": {result.message}" if result.message else ".")
                ),
                details=result.details,
            )

        return CheckResult.success(
            message=(
                "Check correctly failed (inverted)"
                + (f": {result.message}" if result.message else ".")
            ),
            details=result.details,
        )
