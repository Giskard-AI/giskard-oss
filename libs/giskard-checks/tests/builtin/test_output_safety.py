"""Tests for the XSSOutputCheck."""

import pytest
from giskard.checks import Check, CheckStatus, Interaction, Trace, XSSOutputCheck
from giskard.checks.core.extraction import NoMatch
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Clean outputs — should pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outputs",
    [
        "The capital of France is Paris.",
        "Here is a Python function: def add(a, b): return a + b",
        "Use <strong>bold</strong> and <em>italic</em> for emphasis.",
        "",
        42,
        None,
    ],
)
async def test_clean_output_passes(outputs: object) -> None:
    check = XSSOutputCheck()
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.message == "No XSS payload patterns detected in output."


# ---------------------------------------------------------------------------
# XSS payloads — should fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outputs", "expected_label"),
    [
        ("<script>alert(1)</script>", "script tag"),
        ('<SCRIPT SRC="http://evil.example/x.js"></SCRIPT>', "script tag"),
        ('Click <a href="javascript:void(0)">here</a>', "javascript: URI"),
        ("javascript:alert(document.domain)", "javascript: URI"),
        ('<img src=x onerror=alert(1)>', "event handler attribute"),
        ('<body onload=alert(1)>', "event handler attribute"),
        ('<svg onclick=alert(1)>', "event handler attribute"),
        ("eval(atob('YWxlcnQoMSk='))", "eval() call"),
        ("x=eval;x('alert(1)')", "eval() call"),
        ("steal(document.cookie)", "document.cookie access"),
        ("var c = document  .  cookie;", "document.cookie access"),
        ('src="data:text/html,<script>alert(1)</script>"', "data: URI with script"),
    ],
)
async def test_xss_output_fails(outputs: str, expected_label: str) -> None:
    check = XSSOutputCheck()
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs=outputs)
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "XSS payload" in result.message
    assert result.details["matched_pattern"] == expected_label
    assert result.details["matched_text"] is not None


# ---------------------------------------------------------------------------
# Details metadata
# ---------------------------------------------------------------------------


async def test_failure_details_contain_matched_text() -> None:
    check = XSSOutputCheck()
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs="<script>alert(1)</script>")
    )

    result = await check.run(trace)

    assert result.failed
    assert result.details["matched_pattern"] == "script tag"
    assert "<script" in result.details["matched_text"].lower()
    assert result.details["text"] == "<script>alert(1)</script>"


async def test_pass_details_contain_text() -> None:
    check = XSSOutputCheck()
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs="Safe answer.")
    )

    result = await check.run(trace)

    assert result.passed
    assert result.details["text"] == "Safe answer."
    assert "matched_pattern" not in result.details


# ---------------------------------------------------------------------------
# Missing key handling
# ---------------------------------------------------------------------------


async def test_missing_key_fails() -> None:
    check = XSSOutputCheck(key="trace.last.outputs.missing")
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs={"response": "safe"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert isinstance(result.details["value"], NoMatch)
    assert "trace.last.outputs.missing" in (result.message or "")


# ---------------------------------------------------------------------------
# Custom key extraction
# ---------------------------------------------------------------------------


async def test_custom_key_extraction_passes() -> None:
    check = XSSOutputCheck(key="trace.last.outputs.html")
    trace = await Trace.from_interactions(
        Interaction(inputs="Question", outputs={"html": "<p>Hello</p>"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS


async def test_custom_key_extraction_fails() -> None:
    check = XSSOutputCheck(key="trace.last.outputs.html")
    trace = await Trace.from_interactions(
        Interaction(
            inputs="Question",
            outputs={"html": "<p><script>alert(1)</script></p>"},
        )
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.details["matched_pattern"] == "script tag"


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_xss_output_check_is_exported() -> None:
    assert XSSOutputCheck.__name__ == "XSSOutputCheck"


def test_xss_output_check_serialization_roundtrip() -> None:
    check = XSSOutputCheck(key="trace.last.outputs.body")

    data = check.model_dump()
    restored = Check.model_validate(data)

    assert data["kind"] == "xss_output"
    assert isinstance(restored, XSSOutputCheck)
    assert restored.key == "trace.last.outputs.body"


def test_default_key_is_trace_last_outputs() -> None:
    check = XSSOutputCheck()
    assert check.key == "trace.last.outputs"
