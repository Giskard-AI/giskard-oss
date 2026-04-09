from __future__ import annotations

import json
from pathlib import Path

import pytest

from giskard.checks.cli import main


def _write_target_module(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "def explain_python(inputs):",
                '    return "Python is a programming language."',
                "",
                "def echo(inputs):",
                "    return inputs",
            ]
        ),
        encoding="utf-8",
    )


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


def test_validate_scenario_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_target_module(tmp_path / "targets.py")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: basic_test",
                "target: python:targets.explain_python",
                "steps:",
                "  - interact:",
                '      inputs: "What is Python?"',
                "    check:",
                "      - kind: string_matching",
                '        keyword: "programming language"',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(scenario_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid scenario" in captured.out
    assert "basic_test" in captured.out


def test_run_scenario_yaml_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_target_module(tmp_path / "targets.py")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: basic_test",
                "target: python:targets.explain_python",
                "steps:",
                "  - interact:",
                '      inputs: "What is Python?"',
                "    check:",
                "      - kind: string_matching",
                '        keyword: "programming language"',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(scenario_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["scenario_name"] == "basic_test"
    assert payload["status"] == "pass"


def test_run_suite_yaml_writes_junit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_target_module(tmp_path / "targets.py")
    suite_path = tmp_path / "suite.yaml"
    output_path = tmp_path / "results.xml"
    suite_path.write_text(
        "\n".join(
            [
                "name: cli_suite",
                "target: python:targets.echo",
                "scenarios:",
                "  - name: suite_case",
                "    steps:",
                "      - interact:",
                '          inputs: "hello"',
                "        check:",
                "          - kind: equals",
                '            key: "trace.last.outputs"',
                '            expected_value: "hello"',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            str(suite_path),
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "trace_types.py").write_text(
        "\n".join(
            [
                "from giskard.checks import Trace",
                "",
                "class CustomTrace(Trace[str, str]):",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: custom_trace_test",
                "trace_type: python:trace_types.CustomTrace",
                "steps: []",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(scenario_path)])

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
    cached_scenario_path = cwd_dir / "cached.yaml"
    cached_scenario_path.write_text(
        "\n".join(
            [
                "name: cached_target",
                "target: python:targets.explain_python",
                "steps: []",
            ]
        ),
        encoding="utf-8",
    )

    scenario_dir = tmp_path / "scenario_case"
    scenario_dir.mkdir()
    _write_target_module(scenario_dir / "targets.py")
    scenario_path = scenario_dir / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: basic_test",
                "target: python:targets.explain_python",
                "steps:",
                "  - interact:",
                '      inputs: "What is Python?"',
                "    check:",
                "      - kind: string_matching",
                '        keyword: "programming language"',
            ]
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
    assert payload["final_trace"]["last"]["outputs"] == "Python is a programming language."


def test_validate_reports_invalid_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "invalid.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: broken_test",
                "target: python:targets.missing",
                "steps:",
                "  - interact:",
                '      inputs: "hello"',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate", str(scenario_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Could not import module 'targets'" in captured.err
