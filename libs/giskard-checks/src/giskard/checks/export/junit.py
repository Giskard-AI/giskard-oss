import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rich.console import Console

from ..core.result import CheckResult, ScenarioResult, SuiteResult


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


def _iter_checks(scenario: ScenarioResult[Any]):
    for step_index, step in enumerate(scenario.steps, start=1):
        for check_index, check in enumerate(step.results, start=1):
            yield step_index, check_index, check


def _scenario_assertions(scenario: ScenarioResult[Any]) -> int:
    return sum(len(step.results) for step in scenario.steps)


def _suite_assertions(result: SuiteResult) -> int:
    return sum(_scenario_assertions(scenario) for scenario in result.results)


def _suite_counts(result: SuiteResult) -> tuple[int, int, int, int]:
    tests = len(result.results)
    failures = sum(1 for scenario in result.results if scenario.failed)
    errors = sum(1 for scenario in result.results if scenario.errored)
    skipped = sum(1 for scenario in result.results if scenario.skipped)
    return tests, failures, errors, skipped


def _matching_checks(
    scenario: ScenarioResult[Any],
    *,
    failed: bool = False,
    errored: bool = False,
    skipped: bool = False,
) -> list[tuple[int, int, CheckResult]]:
    matches: list[tuple[int, int, CheckResult]] = []
    for step_index, check_index, check in _iter_checks(scenario):
        if failed and check.failed:
            matches.append((step_index, check_index, check))
        elif errored and check.errored:
            matches.append((step_index, check_index, check))
        elif skipped and check.skipped:
            matches.append((step_index, check_index, check))
    return matches


def _build_detail_text(
    scenario: ScenarioResult[Any],
    matches: list[tuple[int, int, CheckResult]],
) -> str:
    lines = [
        f"scenario={scenario.scenario_name}",
        "See testcase properties for final_trace and step payloads.",
    ]

    for step_index, check_index, check in matches:
        label = _check_label(check, f"check_{check_index}")
        status = check.status.value.upper()
        message = check.message or ""
        lines.append(f"[{status}] step_{step_index}.{label}: {message}")
        if check.details:
            lines.append(f"details={_to_json(check.details)}")

    return "\n".join(lines)


def _append_properties(testcase_el: ET.Element, scenario: ScenarioResult[Any]) -> None:
    properties_el = ET.SubElement(testcase_el, "properties")

    ET.SubElement(
        properties_el,
        "property",
        {"name": "final_trace", "value": _to_json(scenario.final_trace)},
    )

    for step_index, step in enumerate(scenario.steps, start=1):
        ET.SubElement(
            properties_el,
            "property",
            {"name": f"step_{step_index}", "value": _to_json(step)},
        )

        for check_index, check in enumerate(step.results, start=1):
            for metric in check.metrics:
                ET.SubElement(
                    properties_el,
                    "property",
                    {
                        "name": f"step_{step_index}.check_{check_index}.{metric.name}",
                        "value": str(metric.value),
                    },
                )


def _append_status_node(testcase_el: ET.Element, scenario: ScenarioResult[Any]) -> None:
    if scenario.errored:
        matches = _matching_checks(scenario, errored=True)
        first = matches[0][2] if matches else None
        node = ET.SubElement(
            testcase_el,
            "error",
            {
                "type": _check_label(first, "error") if first else "error",
                "message": first.message
                if first and first.message
                else "Scenario errored.",
            },
        )
        node.text = _build_detail_text(scenario, matches)
        return

    if scenario.failed:
        matches = _matching_checks(scenario, failed=True)
        first = matches[0][2] if matches else None
        node = ET.SubElement(
            testcase_el,
            "failure",
            {
                "type": _check_label(first, "failure") if first else "failure",
                "message": first.message
                if first and first.message
                else "Scenario failed.",
            },
        )
        node.text = _build_detail_text(scenario, matches)
        return

    if scenario.skipped:
        matches = _matching_checks(scenario, skipped=True)
        node = ET.SubElement(testcase_el, "skipped")
        node.text = _build_detail_text(scenario, matches)


def _render_scenario_report(scenario: ScenarioResult[Any]) -> str:
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        no_color=True,
        width=120,
    )
    scenario.print_report(console)
    return buffer.getvalue().rstrip()


def _append_system_out(testcase_el: ET.Element, scenario: ScenarioResult[Any]) -> None:
    report = _render_scenario_report(scenario)
    if not report:
        return

    system_out = ET.SubElement(testcase_el, "system-out")
    system_out.text = report


def to_junit_xml(result: SuiteResult, path: str | Path | None = None) -> str:
    tests, failures, errors, skipped = _suite_counts(result)

    root = ET.Element(
        "testsuite",
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
        testcase_el = ET.SubElement(
            root,
            "testcase",
            {
                "name": scenario.scenario_name,
                "assertions": str(_scenario_assertions(scenario)),
                "time": _seconds(scenario.duration_ms),
            },
        )

        _append_properties(testcase_el, scenario)
        _append_status_node(testcase_el, scenario)
        _append_system_out(testcase_el, scenario)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return ET.tostring(root, encoding="unicode")
