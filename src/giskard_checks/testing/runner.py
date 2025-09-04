from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from giskard_checks.core.check import CheckResult

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime import cycle
    from giskard_checks.testing.testcase import TestCase


class TestCaseResult(BaseModel):
    model_config = {"frozen": True}
    results: list[CheckResult]
    duration_ms: int

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> bool:
        return not self.errored and any(result.failed for result in self.results)

    @property
    def errored(self) -> bool:
        return any(result.errored for result in self.results)

    @property
    def skipped(self) -> bool:
        return all(result.skipped for result in self.results)


class TestRunner:
    async def run(self, tc: "TestCase[Any]") -> TestCaseResult:
        results: list[CheckResult] = []

        start_time = time.perf_counter()
        for chk in tc.checks:
            check_start_time = time.perf_counter()
            res: CheckResult | None = None
            try:
                res = await chk.run(tc.interaction)
            except Exception as e:
                res = CheckResult.error(
                    kind=chk.kind,
                    name=chk.name,
                    description=chk.description,
                    message=str(e),
                    traceback=traceback.format_exc(),
                )

            # Update the result with the duration, kind, name, and description
            res = res.model_copy(
                update={
                    "duration_ms": int((time.perf_counter() - check_start_time) * 1000),
                    "kind": chk.kind,
                    "name": chk.name,
                    "description": chk.description,
                }
            )

            results.append(res)

        end_time = time.perf_counter()
        total_duration_ms = int((end_time - start_time) * 1000)

        return TestCaseResult(results=results, duration_ms=total_duration_ms)


_default_runner = TestRunner()


def get_runner() -> TestRunner:
    return _default_runner
