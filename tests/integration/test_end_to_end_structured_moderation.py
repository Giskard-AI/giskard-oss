from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from giskard_checks.checks.fn import from_fn
from giskard_checks.core.check import CheckStatus
from giskard_checks.generators import Interaction
from giskard_checks.testing.testcase import TestCase


class ModerationResult(BaseModel):
    moderated: bool
    reason: str | None = None


def make_interaction(moderated: bool, reason: str | None = None) -> Interaction:
    return Interaction(
        inputs="some user text",
        outputs=ModerationResult(moderated=moderated, reason=reason),
    )


async def test_single_pass_boolean_fncheck():
    interaction = make_interaction(moderated=False)

    chk = from_fn(
        lambda inter: not inter.outputs.moderated
        if inter.outputs is not None
        else False,
        name="not_moderated",
        success_message="content is not moderated",
        failure_message="content was moderated",
    )

    tc = TestCase(interaction=interaction, checks=[chk], name="tc-pass")
    result = await tc.run()

    assert len(result.results) == 1
    r = result.results[0]
    assert r.status == CheckStatus.PASS
    assert r.message == "content is not moderated"
    assert r.details.get("check_kind") == "fn"
    assert r.details.get("check_name") == "not_moderated"
    assert (
        isinstance(r.details.get("duration_ms"), int) and r.details["duration_ms"] >= 0
    )

    assert result.passed is True
    assert result.failed is False
    assert result.errored is False


async def test_single_fail_boolean_fncheck():
    interaction = make_interaction(moderated=False)

    chk = from_fn(
        lambda inter: inter.outputs.moderated if inter.outputs is not None else False,
        name="is_moderated",
        failure_message="expected moderated but was not",
    )

    tc = TestCase(interaction=interaction, checks=[chk], name="tc-fail")
    result = await tc.run()

    assert len(result.results) == 1
    r = result.results[0]
    assert r.status == CheckStatus.FAIL
    assert r.message == "expected moderated but was not"
    assert r.details.get("check_kind") == "fn"
    assert r.details.get("check_name") == "is_moderated"
    assert (
        isinstance(r.details.get("duration_ms"), int) and r.details["duration_ms"] >= 0
    )

    assert result.passed is False
    assert result.failed is True
    assert result.errored is False


async def test_single_error_fncheck():
    interaction = make_interaction(moderated=False)

    def boom(_inter):
        raise RuntimeError("boom")

    chk = from_fn(boom, name="boom")

    tc = TestCase(interaction=interaction, checks=[chk], name="tc-error")
    result = await tc.run()

    assert len(result.results) == 1
    r = result.results[0]
    assert r.status == CheckStatus.ERROR
    assert r.message == "Check 'boom' failed with error: boom"
    assert (
        isinstance(r.details.get("traceback"), str)
        and "RuntimeError" in r.details["traceback"]
    )
    assert r.details.get("check_kind") == "fn"
    assert r.details.get("check_name") == "boom"
    assert (
        isinstance(r.details.get("duration_ms"), int) and r.details["duration_ms"] >= 0
    )

    assert result.passed is False
    assert result.failed is False
    assert result.errored is True


async def test_multiple_checks_aggregation_no_error():
    interaction = make_interaction(moderated=False)

    pass_check = from_fn(lambda inter: True, name="pass")
    fail_check = from_fn(lambda inter: False, name="fail")

    tc = TestCase(
        interaction=interaction, checks=[pass_check, fail_check], name="tc-agg-no-error"
    )
    result = await tc.run()

    statuses = [r.status for r in result.results]
    assert statuses == [CheckStatus.PASS, CheckStatus.FAIL]

    assert result.passed is False
    assert result.failed is True
    assert result.errored is False


async def test_multiple_checks_aggregation_with_error():
    interaction = make_interaction(moderated=False)

    pass_check = from_fn(lambda inter: True, name="pass")

    def err(_inter):
        raise ValueError("bad input")

    error_check = from_fn(err, name="error")
    fail_check = from_fn(lambda inter: False, name="fail")

    tc = TestCase(
        interaction=interaction,
        checks=[pass_check, fail_check, error_check],
        name="tc-agg-with-error",
    )
    result = await tc.run()

    statuses = [r.status for r in result.results]
    assert statuses == [CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.ERROR]

    # Any error makes the suite "errored"; "failed" is false when there are any errors
    assert result.passed is False
    assert result.failed is False
    assert result.errored is True
