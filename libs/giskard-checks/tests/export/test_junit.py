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

    assert root.tag == "testsuites"
    assert root.attrib["name"] == "Test run"
    assert root.attrib["tests"] == "4"
    assert root.attrib["failures"] == "1"
    assert root.attrib["errors"] == "1"
    assert root.attrib["skipped"] == "1"
    assert root.attrib["assertions"] == "6"
    assert root.attrib["time"] == "0.420000"
    assert "timestamp" in root.attrib

    testsuites = root.findall("testsuite")
    assert [testsuite.attrib["name"] for testsuite in testsuites] == [
        "scenario_pass",
        "scenario_fail",
        "scenario_error",
        "scenario_skip",
    ]
    assert [testsuite.attrib["tests"] for testsuite in testsuites] == [
        "1",
        "1",
        "1",
        "1",
    ]

    testcases = root.findall("./testsuite/testcase")
    assert [testcase.attrib["name"] for testcase in testcases] == [
        "step_1",
        "step_1",
        "step_1",
        "step_1",
    ]
    assert [testcase.attrib["classname"] for testcase in testcases] == [
        "scenario_pass",
        "scenario_fail",
        "scenario_error",
        "scenario_skip",
    ]


def test_to_junit_xml_includes_properties_and_metrics() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    testsuites = {
        testsuite.attrib["name"]: testsuite for testsuite in root.findall("testsuite")
    }
    pass_suite = testsuites["scenario_pass"]

    suite_properties = pass_suite.find("properties")
    assert suite_properties is not None
    suite_property_map = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in suite_properties.findall("property")
    }
    assert "final_trace" in suite_property_map
    assert suite_property_map["status"] == "pass"

    pass_case = pass_suite.find("testcase")
    assert pass_case is not None

    properties = pass_case.find("properties")
    assert properties is not None

    property_map = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in properties.findall("property")
    }

    assert "result" in property_map
    assert property_map["status"] == "pass"
    assert property_map["check_1.name"] == "Groundedness"
    assert property_map["check_1.status"] == "pass"
    assert property_map["check_1.score"] == "0.95"


def test_to_junit_xml_maps_failure_error_and_skip() -> None:
    xml_string = to_junit_xml(_sample_suite_result())
    root = ET.fromstring(xml_string)

    testcases = {
        testcase.attrib["classname"]: testcase
        for testcase in root.findall("./testsuite/testcase")
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
    assert ET.fromstring(xml_string).tag == "testsuites"
    assert ET.parse(output_path).getroot().tag == "testsuites"


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
                                status=CheckStatus.FAIL,
                                message="second failure",
                                details={"check_name": "CheckB"},
                            ),
                            CheckResult(
                                status=CheckStatus.SKIP,
                                message="skipped follow-up",
                                details={"check_name": "CheckC"},
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
    testcase = root.find("./testsuite/testcase")

    assert testcase is not None
    failure = testcase.find("failure")
    assert failure is not None
    assert len(testcase.findall("failure")) == 1
    assert len(testcase.findall("error")) == 0
    assert len(testcase.findall("skipped")) == 0
    assert "hard failure" in (failure.text or "")
    assert "second failure" in (failure.text or "")
    assert "skipped follow-up" in (failure.text or "")

    testsuite = root.find("testsuite")
    assert testsuite is not None
    assert testsuite.attrib["tests"] == "1"
    assert testsuite.attrib["failures"] == "1"
    assert testsuite.attrib["skipped"] == "0"
