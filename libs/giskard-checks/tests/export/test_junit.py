from pathlib import Path
from xml.etree import ElementTree as ET

from giskard.checks import (
    CheckResult,
    CheckStatus,
    Metric,
    ScenarioResult,
    SuiteResult,
    Trace,
)
from giskard.checks.export.junit import to_junit_xml


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
                                metrics=[Metric(name="score", value=0.95)],
                                details={"check_name": "Groundedness"},
                            ),
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="relevant",
                                details={"check_name": "AnswerRelevance"},
                            ),
                        ],
                        duration_ms=100,
                    )
                ],
                duration_ms=100,
                final_trace=Trace(),
            ),
            ScenarioResult(
                scenario_name="scenario_fail",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="pre-check ok",
                                details={"check_name": "SanityCheck"},
                            ),
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="answer is not grounded",
                                metrics=[Metric(name="confidence", value=0.2)],
                                details={"check_name": "Groundedness"},
                            ),
                        ],
                        duration_ms=120,
                    )
                ],
                duration_ms=120,
                final_trace=Trace(),
            ),
            ScenarioResult(
                scenario_name="scenario_error",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.ERROR,
                                message="judge crashed",
                                details={"check_name": "LLMJudge"},
                            )
                        ],
                        duration_ms=150,
                    )
                ],
                duration_ms=150,
                final_trace=Trace(),
            ),
            ScenarioResult(
                scenario_name="scenario_skip",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="no retrieved context",
                                details={"check_name": "ContextRelevance"},
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
    )


def test_to_junit_xml_builds_valid_xml() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    assert root.tag == "testsuite"
    assert root.attrib["name"] == "Test run"
    assert root.attrib["tests"] == "4"
    assert root.attrib["failures"] == "1"
    assert root.attrib["errors"] == "1"
    assert root.attrib["skipped"] == "1"
    assert root.attrib["assertions"] == "6"
    assert root.attrib["time"] == "0.420000"
    assert "timestamp" in root.attrib

    testcases = root.findall("testcase")
    assert [testcase.attrib["name"] for testcase in testcases] == [
        "scenario_pass",
        "scenario_fail",
        "scenario_error",
        "scenario_skip",
    ]


def test_to_junit_xml_includes_properties_and_metrics() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    testcases = {
        testcase.attrib["name"]: testcase for testcase in root.findall("testcase")
    }
    pass_case = testcases["scenario_pass"]

    properties = pass_case.find("properties")
    assert properties is not None

    property_map = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in properties.findall("property")
    }

    assert "final_trace" in property_map
    assert "step_1" in property_map
    assert property_map["step_1.check_1.score"] == "0.95"


def test_to_junit_xml_maps_failure_error_and_skip() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    testcases = {
        testcase.attrib["name"]: testcase for testcase in root.findall("testcase")
    }

    failure = testcases["scenario_fail"].find("failure")
    assert failure is not None
    assert failure.attrib["message"] == "answer is not grounded"
    assert "answer is not grounded" in (failure.text or "")

    error = testcases["scenario_error"].find("error")
    assert error is not None
    assert error.attrib["message"] == "judge crashed"
    assert "judge crashed" in (error.text or "")

    skipped = testcases["scenario_skip"].find("skipped")
    assert skipped is not None
    assert "no retrieved context" in (skipped.text or "")


def test_to_junit_xml_writes_file(tmp_path: Path) -> None:
    suite_result = _sample_suite_result()
    output_path = tmp_path / "test-results.xml"

    xml_string = to_junit_xml(suite_result, path=output_path)

    assert output_path.exists()
    assert ET.fromstring(xml_string).tag == "testsuite"
    assert ET.parse(output_path).getroot().tag == "testsuite"


def test_suite_result_convenience_method_matches_function() -> None:
    suite_result = _sample_suite_result()

    xml_from_method = suite_result.to_junit_xml()
    xml_from_function = to_junit_xml(suite_result)

    root_from_method = ET.fromstring(xml_from_method)
    root_from_function = ET.fromstring(xml_from_function)

    assert root_from_method.tag == root_from_function.tag
    assert root_from_method.attrib["tests"] == root_from_function.attrib["tests"]
    assert (
        root_from_method.attrib["assertions"] == root_from_function.attrib["assertions"]
    )


def test_failed_scenario_with_mixed_check_statuses_is_still_a_failure() -> None:
    from giskard.checks import TestCaseResult

    suite_result = SuiteResult(
        results=[
            ScenarioResult(
                scenario_name="scenario_mixed",
                steps=[
                    TestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="hard failure",
                                details={"check_name": "CheckA"},
                            ),
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="skipped follow-up",
                                details={"check_name": "CheckB"},
                            ),
                        ],
                        duration_ms=50,
                    )
                ],
                duration_ms=50,
                final_trace=Trace(),
            )
        ],
        duration_ms=50,
    )

    root = ET.fromstring(to_junit_xml(suite_result))
    testcase = root.find("testcase")

    assert testcase is not None
    assert testcase.find("failure") is not None
    assert testcase.find("skipped") is None

    system_out = testcase.find("system-out")
    assert system_out is not None
    assert system_out.text is not None
    assert system_out.text.strip() != ""
    assert "final_trace=" not in system_out.text
    assert "step_1=" not in system_out.text
