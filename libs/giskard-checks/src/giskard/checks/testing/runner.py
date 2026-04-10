import time
import traceback

from giskard.core import scoped_telemetry, telemetry, telemetry_tag

from .._telemetry_props import (
    check_kind_counts_from_sequence,
    test_case_shape_properties,
)
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

    @scoped_telemetry
    async def run[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        test_case: TestCase[InputType, OutputType, TraceType],
        return_exception: bool = False,
    ) -> TestCaseResult:
        telemetry_tag("giskard_component", "test_case_runner")
        telemetry_tag("giskard_operation", "test_case_run")

        start_time = time.perf_counter()
        checks_list = list(test_case.checks)
        check_kinds = check_kind_counts_from_sequence(checks_list)
        trace = test_case.trace

        shape_props = test_case_shape_properties(
            check_count=len(checks_list),
            trace_interaction_count=len(trace.interactions),
            has_trace_annotations=bool(trace.annotations),
            has_test_case_name=test_case.name is not None,
            check_kinds=check_kinds,
        )
        _ = telemetry.capture(
            "checks_test_case_run_started",
            properties=shape_props,
        )

        check_results: list[CheckResult] = []
        for check in checks_list:
            result = await _run_check(test_case.trace, check, return_exception)
            check_results.append(result)

        end_time = time.perf_counter()
        total_duration_ms = int((end_time - start_time) * 1000)

        tc_result = TestCaseResult(
            results=check_results,
            duration_ms=total_duration_ms,
        )

        _ = telemetry.capture(
            "checks_test_case_run_finished",
            properties={
                **shape_props,
                "outcome_status": tc_result.status.value,
                "duration_ms": total_duration_ms,
                "return_exception_mode": return_exception,
            },
        )

        return tc_result


_default_runner = TestCaseRunner()


def get_runner() -> TestCaseRunner:
    return _default_runner
