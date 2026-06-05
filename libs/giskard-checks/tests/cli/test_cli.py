import json
from pathlib import Path

import pytest
from giskard.checks.cli import main

DATA_DIR = Path(__file__).parent / "data"


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "validate" in captured.out
    assert "list" in captured.out


def test_list_checks_includes_string_matching(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["list", "checks"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "string_matching" in captured.out
    assert "regex_matching" in captured.out


def test_validate_scenario_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", str(DATA_DIR / "scenario.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid scenario" in captured.out
    assert "basic_test" in captured.out


def test_validate_scenario_yaml_when_pyyaml_is_available(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("yaml")

    exit_code = main(["validate", str(DATA_DIR / "scenario.yaml")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid scenario" in captured.out
    assert "basic_test" in captured.out


def test_run_scenario_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", str(DATA_DIR / "scenario.json"), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["scenario_name"] == "basic_test"
    assert payload["status"] == "pass"


def test_run_suite_json_writes_junit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "reports" / "results.xml"

    exit_code = main(
        [
            "run",
            str(DATA_DIR / "suite.json"),
            "--format",
            "junit",
            "--output",
            str(output_path),
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert "<testsuite" in output_path.read_text(encoding="utf-8")


def test_validate_scenario_with_trace_type_string(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["validate", str(DATA_DIR / "trace_type.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid scenario" in captured.out
    assert "custom_trace_test" in captured.out


def test_run_prefers_definition_directory_targets_over_cwd_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd_dir = tmp_path / "cwd_case"
    cwd_dir.mkdir()
    (cwd_dir / "targets.py").write_text(
        "\n".join(
            [
                "def explain_python(inputs):",
                '    return "Wrong answer from cwd."',
            ]
        ),
        encoding="utf-8",
    )
    cached_scenario_path = cwd_dir / "cached.json"
    cached_scenario_path.write_text(
        json.dumps(
            {
                "name": "cached_target",
                "target": "python:targets.explain_python",
                "steps": [],
            }
        ),
        encoding="utf-8",
    )

    scenario_dir = tmp_path / "scenario_case"
    scenario_dir.mkdir()
    (scenario_dir / "targets.py").write_text(
        "\n".join(
            [
                "def explain_python(inputs):",
                '    return "Python is a programming language."',
            ]
        ),
        encoding="utf-8",
    )
    scenario_path = scenario_dir / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "basic_test",
                "target": "python:targets.explain_python",
                "steps": [
                    {
                        "interacts": [
                            {"kind": "interact", "inputs": "What is Python?"}
                        ],
                        "checks": [
                            {
                                "kind": "string_matching",
                                "keyword": "programming language",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(cwd_dir)
    assert main(["validate", str(cached_scenario_path)]) == 0
    capsys.readouterr()

    exit_code = main(["run", str(scenario_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert (
        payload["final_trace"]["last"]["outputs"] == "Python is a programming language."
    )


def test_validate_reports_invalid_target(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", str(DATA_DIR / "invalid_target.json")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Could not import module 'missing_targets'" in captured.err
