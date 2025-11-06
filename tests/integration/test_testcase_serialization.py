from __future__ import annotations

from typing import Any

import pytest
from giskard.checks.core import Check, CheckResult, CheckStatus, Interaction
from giskard.checks.testing.testcase import TestCase


@Check.register("contains")
class ContainsCheck(Check):
    needle: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
        if interaction.inputs is not None and self.needle in interaction.inputs:
            return CheckResult.success(
                message=f"input contains '{self.needle}'",
            )
        return CheckResult.failure(
            message=f"input does not contain '{self.needle}'",
        )


@Check.register("equals")
class OutputEqualsCheck(Check):
    expected: str

    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
        if interaction.outputs == self.expected:
            return CheckResult.success(
                message="output matched",
            )
        return CheckResult.failure(
            message=f"expected '{self.expected}', got '{interaction.outputs}'",
        )


@Check.register("explode")
class ExplodeCheck(Check):
    async def run(self, interaction: Interaction[Any, Any]) -> CheckResult:
        raise RuntimeError("kaboom")


async def test_testcase_roundtrip_serialization_custom_checks():
    interaction = Interaction(inputs="hello world", outputs="hello")

    chk1 = ContainsCheck(name="has_hello", needle="hello")
    chk2 = OutputEqualsCheck(name="out_is_hello", expected="hello")
    chk3 = ExplodeCheck(name="explode")

    tc = TestCase(
        name="tc-serialize",
        interaction=interaction,
        checks=[chk1, chk2, chk3],
    )

    # Run before serialization
    before = await tc.run()
    before_statuses = [r.status for r in before.results]
    assert before_statuses == [CheckStatus.PASS, CheckStatus.PASS, CheckStatus.ERROR]

    # Serialize and reconstruct
    payload = tc.model_dump()
    tc2 = TestCase.model_validate(payload)

    # Structural equivalence
    assert tc2.name == tc.name
    assert tc2.interaction.model_dump() == tc.interaction.model_dump()
    assert [c.kind for c in tc2.checks] == ["contains", "equals", "explode"]

    # Ensure subclass fields round-trip
    assert isinstance(tc2.checks[0], ContainsCheck)
    assert isinstance(tc2.checks[1], OutputEqualsCheck)
    assert isinstance(tc2.checks[2], ExplodeCheck)
    assert getattr(tc2.checks[0], "needle") == "hello"
    assert getattr(tc2.checks[1], "expected") == "hello"

    # Run after deserialization and compare outcomes (ignoring durations)
    after = await tc2.run()
    after_statuses = [r.status for r in after.results]
    assert after_statuses == before_statuses


def test_deserialize_rejects_check_without_kind():
    payload: dict[str, Any] = {
        "name": "tc-bad-missing-kind",
        "interaction": {
            "inputs": "hello",
            "outputs": None,
            "metadata": None,
        },
        "checks": [
            {
                # no 'kind' field
            }
        ],
    }

    with pytest.raises(ValueError) as err:
        TestCase.model_validate(payload)

    assert "Kind is not provided for" in str(err.value)


def test_deserialize_rejects_check_with_empty_kind():
    payload: dict[str, Any] = {
        "name": "tc-bad-empty-kind",
        "interaction": {
            "input": "hello",
            "output": None,
            "metadata": None,
        },
        "checks": [
            {
                "kind": "",
            }
        ],
    }

    with pytest.raises(ValueError) as err:
        TestCase.model_validate(payload)

    assert "Kind" in str(err.value) and "not registered" in str(err.value)
