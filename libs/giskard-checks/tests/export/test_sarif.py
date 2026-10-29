import json
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema
import pytest
from giskard.checks import (
    CheckResult,
    CheckStatus,
    Metric,
    ScenarioResult,
    SuiteResult,
    Trace,
)
from giskard.checks.export.sarif import to_sarif
from giskard.checks.scenarios.suite import Suite

SARIF_SCHEMA_URL = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
    "sarif-2.1/schema/sarif-schema-2.1.0.json"
)


def _sample_suite_result() -> SuiteResult:
    from giskard.checks import TestCaseResult

    return SuiteResult(
        results=[
            ScenarioResult(
                scenario_name="scenario_pass",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="grounded",
                                details={
                                    "check_name": "Groundedness",
                                    "check_kind": "groundedness",
                                },
                            ),
                        ],
                        duration_ms=100,
                    )
                ],
                duration_ms=100,
                final_trace=Trace(),
                tags=["threat-type:hallucination"],
            ),
            ScenarioResult(
                scenario_name="scenario_fail",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="answer is not grounded",
                                metrics=[Metric(name="confidence", value=0.2)],
                                details={
                                    "check_name": "Groundedness",
                                    "check_kind": "groundedness",
                                },
                            )
                        ],
                        duration_ms=120,
                    )
                ],
                duration_ms=120,
                final_trace=Trace(),
                tags=[
                    "threat-type:hallucination",
                    "owasp:llm-top-10-2025:LLM09",
                ],
            ),
            ScenarioResult(
                scenario_name="scenario_error",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.ERROR,
                                message="judge crashed",
                                details={
                                    "check_name": "LLMJudge",
                                    "check_kind": "llm_judge",
                                },
                            )
                        ],
                        duration_ms=150,
                    )
                ],
                duration_ms=150,
                final_trace=Trace(),
                tags=["threat-type:prompt-injection"],
            ),
            ScenarioResult(
                scenario_name="scenario_skip",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="no retrieved context",
                                details={
                                    "check_name": "ContextRelevance",
                                    "check_kind": "context_relevance",
                                },
                            )
                        ],
                        duration_ms=50,
                    )
                ],
                duration_ms=50,
                final_trace=Trace(),
            ),
        ],
        duration_ms=420,
        suite=Suite(name="test"),
    )


def test_to_sarif_only_reports_failures_and_errors() -> None:
    document = json.loads(to_sarif(_sample_suite_result()))

    assert document["version"] == "2.1.0"
    assert "$schema" in document
    assert len(document["runs"]) == 1
    run = document["runs"][0]

    assert run["tool"]["driver"]["name"] == "giskard-checks"

    result_rule_ids = [r["ruleId"] for r in run["results"]]
    assert result_rule_ids == ["groundedness", "llm_judge"]


def test_to_sarif_maps_fail_to_error_and_error_to_warning() -> None:
    document = json.loads(to_sarif(_sample_suite_result()))
    results_by_scenario = {
        r["properties"]["scenario"]: r for r in document["runs"][0]["results"]
    }

    assert results_by_scenario["scenario_fail"]["level"] == "error"
    assert results_by_scenario["scenario_error"]["level"] == "warning"


def test_to_sarif_deduplicates_rules_by_check_kind() -> None:
    document = json.loads(to_sarif(_sample_suite_result()))
    rules = document["runs"][0]["tool"]["driver"]["rules"]

    rule_ids = [rule["id"] for rule in rules]
    assert rule_ids == ["groundedness", "llm_judge"]
    assert len(rule_ids) == len(set(rule_ids))


def test_to_sarif_unions_scenario_tags_onto_rule() -> None:
    document = json.loads(to_sarif(_sample_suite_result()))
    rules_by_id = {
        rule["id"]: rule for rule in document["runs"][0]["tool"]["driver"]["rules"]
    }

    assert rules_by_id["groundedness"]["properties"]["tags"] == [
        "owasp:llm-top-10-2025:LLM09",
        "threat-type:hallucination",
    ]
    assert rules_by_id["llm_judge"]["properties"]["tags"] == [
        "threat-type:prompt-injection"
    ]


def test_to_sarif_includes_stable_partial_fingerprint() -> None:
    doc_a = json.loads(to_sarif(_sample_suite_result()))
    doc_b = json.loads(to_sarif(_sample_suite_result()))

    prints_a = [r["partialFingerprints"] for r in doc_a["runs"][0]["results"]]
    prints_b = [r["partialFingerprints"] for r in doc_b["runs"][0]["results"]]
    assert prints_a == prints_b
    assert all("giskardCheck/v1" in p for p in prints_a)


def test_to_sarif_writes_file(tmp_path: Path) -> None:
    output_path = tmp_path / "giskard.sarif"
    serialized = to_sarif(_sample_suite_result(), path=output_path)

    assert output_path.exists()
    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert json.loads(serialized) == on_disk


def test_suite_result_convenience_method_matches_function() -> None:
    suite_result = _sample_suite_result()

    from_method = json.loads(suite_result.to_sarif())
    from_function = json.loads(to_sarif(suite_result))

    assert from_method["runs"][0]["results"] == from_function["runs"][0]["results"]


def test_to_sarif_handles_empty_suite_result() -> None:
    empty = SuiteResult(results=[], duration_ms=0, suite=Suite(name="empty"))
    document = json.loads(to_sarif(empty))

    run = document["runs"][0]
    assert run["results"] == []
    assert run["tool"]["driver"]["rules"] == []


def test_to_sarif_omits_metrics_when_absent() -> None:
    document = json.loads(to_sarif(_sample_suite_result()))
    results_by_scenario = {
        r["properties"]["scenario"]: r for r in document["runs"][0]["results"]
    }

    assert results_by_scenario["scenario_fail"]["properties"]["metrics"] == {
        "confidence": 0.2
    }
    assert "metrics" not in results_by_scenario["scenario_error"]["properties"]


@pytest.mark.integration
def test_to_sarif_validates_against_official_schema() -> None:
    try:
        with urllib.request.urlopen(SARIF_SCHEMA_URL, timeout=10) as response:
            schema = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"Could not fetch SARIF schema: {exc}")

    document = json.loads(to_sarif(_sample_suite_result()))
    jsonschema.validate(instance=document, schema=schema)
