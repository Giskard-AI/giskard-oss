"""SARIF 2.1.0 exporter for ``SuiteResult``.

SARIF puts each finding in the GitHub Security tab and as a PR annotation,
with a rule, a level and a stable fingerprint. It answers a different
question than JUnit: not "did the build pass" but "which findings changed
across runs, and are any of them new".

Design choices:

* Only ``FAIL`` and ``ERROR`` checks become SARIF results. ``PASS`` and
  ``SKIP`` are omitted so the Security tab is not swamped by an alert per
  passing check.
* Rules are keyed by ``check_kind`` (falling back to ``check_name``), so a
  single rule covers every instance rather than one rule per scenario.
* Scenario tags become rule tags, unioned across scenarios that use the
  same check. This is where an OWASP or MITRE mapping later slots in.
"""

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from ..core.result import CheckResult, ScenarioResult, SuiteResult

SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Documents/CommitteeSpecificationDrafts/2.1.0/CSD.2.1.0.json"
)
SARIF_VERSION = "2.1.0"
TOOL_NAME = "giskard-checks"
TOOL_INFORMATION_URI = "https://github.com/Giskard-AI/giskard-oss"


def _tool_version() -> str:
    try:
        return metadata.version(TOOL_NAME)
    except metadata.PackageNotFoundError:
        return "unknown"


def _rule_id(check: CheckResult) -> str:
    details = check.details if isinstance(check.details, dict) else {}
    return str(
        details.get("check_kind") or details.get("check_name") or "unknown_check"
    )


def _rule_name(check: CheckResult) -> str:
    details = check.details if isinstance(check.details, dict) else {}
    return str(
        details.get("check_name") or details.get("check_kind") or "Unknown check"
    )


def _sarif_level(check: CheckResult) -> str:
    # FAIL is a security-relevant verdict; ERROR is a tooling problem that
    # did not reach a verdict, so we surface it as a warning rather than
    # an error to keep the two distinct in the Security tab.
    if check.failed:
        return "error"
    return "warning"


def _fingerprint(
    rule_id: str,
    scenario_name: str,
    step_index: int,
    check_index: int,
) -> str:
    digest = hashlib.sha256(
        f"{rule_id}|{scenario_name}|{step_index}|{check_index}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _location(scenario_name: str, step_index: int, check_index: int) -> dict[str, Any]:
    return {
        "logicalLocations": [
            {
                "fullyQualifiedName": (
                    f"{scenario_name}/step_{step_index}/check_{check_index}"
                ),
                "kind": "check",
            }
        ]
    }


def _build_rule(
    rule_id: str,
    rule_name: str,
    tags: list[str],
) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "id": rule_id,
        "name": rule_name,
        "shortDescription": {"text": rule_name},
        "fullDescription": {
            "text": f"Giskard check '{rule_name}' evaluated against a scenario."
        },
        "defaultConfiguration": {"level": "warning"},
    }
    if tags:
        rule["properties"] = {"tags": tags}
    return rule


def _build_result(
    rule_id: str,
    check: CheckResult,
    scenario: ScenarioResult[Any],
    step_index: int,
    check_index: int,
) -> dict[str, Any]:
    message_text = check.message or (
        f"{rule_id} did not pass in scenario {scenario.scenario_name!r}."
    )
    sarif_result: dict[str, Any] = {
        "ruleId": rule_id,
        "level": _sarif_level(check),
        "message": {"text": message_text},
        "locations": [_location(scenario.scenario_name, step_index, check_index)],
        "partialFingerprints": {
            "giskardCheck/v1": _fingerprint(
                rule_id, scenario.scenario_name, step_index, check_index
            )
        },
    }
    properties: dict[str, Any] = {
        "scenario": scenario.scenario_name,
        "status": check.status.value,
    }
    if scenario.tags:
        properties["scenarioTags"] = list(scenario.tags)
    if check.metrics:
        properties["metrics"] = {m.name: m.value for m in check.metrics}
    sarif_result["properties"] = properties
    return sarif_result


def _iter_reportable(result: SuiteResult):
    for scenario in result.results:
        for step_index, step in enumerate(scenario.steps, start=1):
            for check_index, check in enumerate(step.results, start=1):
                if check.failed or check.errored:
                    yield scenario, step_index, check_index, check


def to_sarif(result: SuiteResult, path: str | Path | None = None) -> str:
    """Serialize a ``SuiteResult`` to a SARIF 2.1.0 JSON string.

    Parameters
    ----------
    result : SuiteResult
        The suite result to export.
    path : str or Path, optional
        When provided, the SARIF document is also written to this file
        (parent directories are created).

    Returns
    -------
    str
        The SARIF document as a JSON string.
    """
    rules_by_id: dict[str, dict[str, Any]] = {}
    rule_tags: dict[str, set[str]] = {}
    sarif_results: list[dict[str, Any]] = []

    for scenario, step_index, check_index, check in _iter_reportable(result):
        rule_id = _rule_id(check)
        rule_tags.setdefault(rule_id, set()).update(scenario.tags)
        if rule_id not in rules_by_id:
            rules_by_id[rule_id] = _build_rule(rule_id, _rule_name(check), [])
        sarif_results.append(
            _build_result(rule_id, check, scenario, step_index, check_index)
        )

    for rule_id, rule in rules_by_id.items():
        tags = sorted(rule_tags.get(rule_id, set()))
        if tags:
            rule.setdefault("properties", {})["tags"] = tags

    document: dict[str, Any] = {
        "$schema": SARIF_SCHEMA_URI,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": TOOL_NAME,
                        "version": _tool_version(),
                        "informationUri": TOOL_INFORMATION_URI,
                        "rules": list(rules_by_id.values()),
                    }
                },
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": datetime.now(timezone.utc)
                        .isoformat(timespec="seconds")
                        .replace("+00:00", "Z"),
                    }
                ],
                "results": sarif_results,
            }
        ],
    }

    serialized = json.dumps(document, ensure_ascii=False, indent=2, default=str)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")

    return serialized
