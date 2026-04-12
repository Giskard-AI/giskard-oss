import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.result import CheckResult, ScenarioResult, SuiteResult, TestCaseResult


def _seconds(duration_ms: int) -> str:
    return f"{duration_ms / 1000:.6f}"


def _to_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, default=str)


def _check_label(result: CheckResult, fallback: str) -> str:
    if isinstance(result.details, dict):
        return str(
            result.details.get("check_name")
            or result.details.get("check_kind")
            or result.details.get("name")
            or fallback
        )
    return fallback


def _iter_checks(step: TestCaseResult) -> Iterator[tuple[int, CheckResult]]:
    for check_index, check in enumerate(step.results, start=1):
        yield check_index, check


def _step_assertions(step: TestCaseResult) -> int:
    return len(step.results)


def _scenario_assertions(scenario: ScenarioResult[Any]) -> int:
    return sum(_step_assertions(step) for step in scenario.steps)


def _suite_assertions(result: SuiteResult) -> int:
    return sum(_scenario_assertions(scenario) for scenario in result.results)


def _scenario_counts(scenario: ScenarioResult[Any]) -> tuple[int, int, int, int]:
    tests = len(scenario.steps)
    failures = sum(1 for step in scenario.steps if step.failed)
    errors = sum(1 for step in scenario.steps if step.errored)
    skipped = sum(1 for step in scenario.steps if step.skipped)
    return tests, failures, errors, skipped


def _suite_counts(result: SuiteResult) -> tuple[int, int, int, int]:
    counts = [_scenario_counts(scenario) for scenario in result.results]
    tests = sum(count[0] for count in counts)
    failures = sum(count[1] for count in counts)
    errors = sum(count[2] for count in counts)
    skipped = sum(count[3] for count in counts)
    return tests, failures, errors, skipped


def _build_check_detail_text(
    scenario_name: str,
    step_index: int,
    check_index: int,
    check: CheckResult,
) -> str:
    label = _check_label(check, f"check_{check_index}")
    status = check.status.value.upper()
    message = check.message or ""
    lines = [
        f"scenario={scenario_name}",
        f"testcase=step_{step_index}",
        f"check={label}",
        f"status={status}",
        f"message={message}",
    ]
    if check.details:
        lines.append(f"details={_to_json(check.details)}")

    return "\n".join(lines)


def _build_step_detail_text(
    scenario_name: str,
    step_index: int,
    checks: list[tuple[int, CheckResult]],
) -> str:
    return "\n\n".join(
        _build_check_detail_text(scenario_name, step_index, check_index, check)
        for check_index, check in checks
    )


def _append_scenario_properties(
    testsuite_el: ET.Element, scenario: ScenarioResult[Any]
) -> None:
    properties_el = ET.SubElement(testsuite_el, "properties")
    ET.SubElement(
        properties_el,
        "property",
        {"name": "final_trace", "value": _to_json(scenario.final_trace)},
    )

    ET.SubElement(
        properties_el,
        "property",
        {"name": "status", "value": scenario.status.value},
    )


def _append_testcase_properties(
    testcase_el: ET.Element, step: TestCaseResult
) -> None:
    properties_el = ET.SubElement(testcase_el, "properties")

    ET.SubElement(
        properties_el,
        "property",
        {"name": "status", "value": step.status.value},
    )

    ET.SubElement(
        properties_el,
        "property",
        {"name": "result", "value": _to_json(step)},
    )

    for check_index, check in _iter_checks(step):
        label = _check_label(check, f"check_{check_index}")
        ET.SubElement(
            properties_el,
            "property",
            {"name": f"check_{check_index}.name", "value": label},
        )
        ET.SubElement(
            properties_el,
            "property",
            {"name": f"check_{check_index}.status", "value": check.status.value},
        )

        for metric in check.metrics:
            ET.SubElement(
                properties_el,
                "property",
                {
                    "name": f"check_{check_index}.{metric.name}",
                    "value": str(metric.value),
                },
            )


def _append_status_nodes(
    testcase_el: ET.Element,
    scenario_name: str,
    step_index: int,
    step: TestCaseResult,
) -> None:
    checks = [
        (check_index, check)
        for check_index, check in _iter_checks(step)
        if not check.passed
    ]
    if not checks:
        return

    if step.errored:
        tag = "error"
        default_message = "Test case errored."
        priority_checks = [check for check in checks if check[1].errored]
    elif step.failed:
        tag = "failure"
        default_message = "Test case failed."
        priority_checks = [check for check in checks if check[1].failed]
    else:
        tag = "skipped"
        default_message = "Test case skipped."
        priority_checks = [check for check in checks if check[1].skipped]

    _, first = priority_checks[0] if priority_checks else checks[0]
    node = ET.SubElement(
        testcase_el,
        tag,
        {
            "type": _check_label(first, tag),
            "message": first.message or default_message,
        },
    )
    node.text = _build_step_detail_text(scenario_name, step_index, checks)


def to_junit_xml(result: SuiteResult, path: str | Path | None = None) -> str:
    """Export a suite result as JUnit XML.

    The XML hierarchy follows the Giskard result hierarchy:
    `SuiteResult` -> `testsuites`, `ScenarioResult` -> `testsuite`,
    `TestCaseResult` -> `testcase`, and non-passing `CheckResult` values
    -> `failure`, `error`, or `skipped` nodes.
    """
    tests, failures, errors, skipped = _suite_counts(result)

    root = ET.Element(
        "testsuites",
        {
            "name": "Test run",
            "tests": str(tests),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "assertions": str(_suite_assertions(result)),
            "time": _seconds(result.duration_ms),
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
    )

    for scenario in result.results:
        scenario_tests, scenario_failures, scenario_errors, scenario_skipped = (
            _scenario_counts(scenario)
        )
        testsuite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": scenario.scenario_name,
                "tests": str(scenario_tests),
                "failures": str(scenario_failures),
                "errors": str(scenario_errors),
                "skipped": str(scenario_skipped),
                "assertions": str(_scenario_assertions(scenario)),
                "time": _seconds(scenario.duration_ms),
            },
        )

        _append_scenario_properties(testsuite_el, scenario)

        for step_index, step in enumerate(scenario.steps, start=1):
            testcase_el = ET.SubElement(
                testsuite_el,
                "testcase",
                {
                    "name": f"step_{step_index}",
                    "classname": scenario.scenario_name,
                    "assertions": str(_step_assertions(step)),
                    "time": _seconds(step.duration_ms),
                },
            )

            _append_testcase_properties(testcase_el, step)
            _append_status_nodes(testcase_el, scenario.scenario_name, step_index, step)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return ET.tostring(root, encoding="unicode")
