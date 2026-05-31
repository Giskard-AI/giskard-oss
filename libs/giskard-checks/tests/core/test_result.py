import io
import re

from giskard.checks import (
    CheckResult,
    CheckStatus,
    ScenarioResult,
    SuiteResult,
    TestCaseResult,
    Trace,
)
from rich.console import Console


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _suite_result_with_failures(*, limit: int = 20) -> SuiteResult:
    return SuiteResult(
        results=[
            ScenarioResult(
                scenario_name=f"scenario_{idx}",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message=f"failure {idx}",
                                details={"check_name": f"Check{idx}"},
                            )
                        ],
                        duration_ms=1,
                    )
                ],
                duration_ms=1,
                final_trace=Trace(),
            )
            for idx in range(25)
        ],
        duration_ms=25,
        max_loggable_failures=limit,
    )


def test_suite_result_renders_default_failure_limit(capsys) -> None:
    suite_result = _suite_result_with_failures()

    buffer = io.StringIO()
    Console(file=buffer).print(suite_result)
    output = strip_ansi(buffer.getvalue())

    assert output.count("scenario_") == 40
    assert "... and 5 more" in output


def test_suite_result_renders_custom_failure_limit(capsys) -> None:
    suite_result = _suite_result_with_failures(limit=5)

    buffer = io.StringIO()
    Console(file=buffer).print(suite_result)
    output = strip_ansi(buffer.getvalue())

    assert output.count("scenario_") == 10
    assert "... and 20 more" in output
