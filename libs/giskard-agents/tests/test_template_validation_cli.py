import json
import os
import subprocess
import sys


def _cli(stdin, *args):
    env = {**os.environ, "DO_NOT_TRACK": "1"}
    return subprocess.run(
        [sys.executable, "-I", "-m", "giskard.agents.templates.validate", *args],
        input=stdin,
        env=env,
        capture_output=True,
        text=True,
    )


def test_round_trip_keeps_order():
    result = _cli(
        json.dumps(["{{ trace }}", "{{ trace | nosuch }}", "{{ x }}"]),
        "--variables",
        "trace",
        "--required",
        "trace",
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out[0] == {"ok": True}
    assert out[1]["ok"] is False and "nosuch" in out[1]["error"]
    assert out[2]["ok"] is False and "found: x" in out[2]["error"]
    assert all(set(item) == {"ok", "error", "line"} for item in out[1:])


def test_bad_input_exits_2():
    result = _cli('{"not": "a list"}', "--variables", "trace")
    assert result.returncode == 2
    assert result.stderr.strip()


def test_invalid_json_and_non_string_items_exit_2():
    assert _cli("not json", "--variables", "trace").returncode == 2
    assert _cli("[1]", "--variables", "trace").returncode == 2


def test_empty_array():
    result = _cli("[]", "--variables", "trace")
    assert result.returncode == 0
    assert json.loads(result.stdout) == []


def test_version():
    result = _cli("", "--version")
    assert result.returncode == 0
    assert result.stdout.startswith("giskard-agents ")


def test_deeply_nested_template_does_not_crash_batch():
    deep = "{{" + "(" * 1990 + "1" + ")" * 1990 + "}}{{ trace }}"
    nested = "{% if 1 %}" * 100 + "{{ trace }}" + "{% endif %}" * 100
    result = _cli(json.dumps([deep, "{{ trace }}", nested]), "--variables", "trace")
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert [item["ok"] for item in out] == [False, True, False]
    assert out[0]["error"].startswith("not a valid template:")


def test_required_outside_variables_exits_2():
    result = _cli("[]", "--variables", "trace", "--required", "other")
    assert result.returncode == 2
    assert "other" in result.stderr
