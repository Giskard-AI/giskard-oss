from __future__ import annotations

import time
import traceback

from ..core import Trace
from ..core.check import Check
from ..core.result import CheckResult, TestCaseResult
from ..core.testcase import TestCase


async def _run_check[
    InputType,
    OutputType,
    TraceType: Trace,  # pyright: ignore[reportMissingTypeArgument]
](
    trace: TraceType,
    check: Check[InputType, OutputType, TraceType],
    return_exception: bool = False,
) -> CheckResult:
    check_start_time = time.perf_counter()
    res: CheckResult | None = None

    try:
        res = await check.run(trace)
    except Exception as e:
        if not return_exception:
            raise e
        res = CheckResult.error(
            message=f"Check '{check.name or check.kind}' failed with error: {str(e)}",
            details={
                "traceback": traceback.format_exc(),
                "exception_type": type(e).__name__,
            },
        )

    # Update the result with the duration in details for observability
    return res.model_copy(
        update={
            "details": {
                **(res.details or {}),
                "duration_ms": int((time.perf_counter() - check_start_time) * 1000),
                "check_kind": check.kind,
                "check_name": check.name,
                "check_description": check.description,
            }
        }
    )


class TestCaseRunner:
    def __init__(self):
        pass

    async def run[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        test_case: TestCase[InputType, OutputType, TraceType],
        return_exception: bool = False,
    ) -> TestCaseResult:
        start_time = time.perf_counter()

        check_results: list[CheckResult] = []
        for check in test_case.checks:
            result = await _run_check(test_case.trace, check, return_exception)
            check_results.append(result)

        end_time = time.perf_counter()
        total_duration_ms = int((end_time - start_time) * 1000)

        return TestCaseResult(
            results=check_results,
            duration_ms=total_duration_ms,
        )


_default_runner = TestCaseRunner()


def get_runner() -> TestCaseRunner:
    return _default_runner
