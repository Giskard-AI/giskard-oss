from __future__ import annotations

from typing import Any

import pytest

from giskard_checks.core import Check, CheckResult, CheckStatus
from giskard_checks.interactions import StructuredInteraction
from giskard_checks.testing.testcase import TestCase


class ContainsCheck(Check[StructuredInteraction[str, str]]):
    KIND = "contains"

    needle: str

    async def run(self, interaction: StructuredInteraction[str, str]) -> CheckResult:
        if interaction.input is not None and self.needle in interaction.input:
            return CheckResult.success(
                message=f"input contains '{self.needle}'",
            )
        return CheckResult.failure(
            message=f"input does not contain '{self.needle}'",
        )


class OutputEqualsCheck(Check[StructuredInteraction[str, str]]):
    KIND = "equals"

    expected: str

    async def run(self, interaction: StructuredInteraction[str, str]) -> CheckResult:
        if interaction.output == self.expected:
            return CheckResult.success(
                message="output matched",
            )
        return CheckResult.failure(
            message=f"expected '{self.expected}', got '{interaction.output}'",
        )


class ExplodeCheck(Check[StructuredInteraction[str, str]]):
    KIND = "explode"

    async def run(self, interaction: StructuredInteraction[str, str]) -> CheckResult:  # type: ignore[override]
        raise RuntimeError("kaboom")


async def test_testcase_roundtrip_serialization_custom_checks():
    interaction = StructuredInteraction[str, str](input="hello world", output="hello")

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
    payload = tc.serialize()
    tc2 = TestCase.deserialize(payload)

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


def _minimal_interaction_payload() -> dict[str, Any]:
    return {
        "__type__": "giskard_checks.interactions.structured.StructuredInteraction",
        "data": {"input": "hello", "output": None, "metadata": None},
    }


def test_deserialize_rejects_check_without_kind_and_no_type():
    payload: dict[str, Any] = {
        "name": "tc-bad-missing-kind",
        "interaction": _minimal_interaction_payload(),
        "checks": [
            {
                # no 'kind' and no '__type__'
            }
        ],
    }

    with pytest.raises(ValueError) as err:
        TestCase.deserialize(payload)

    assert "Serialized check must include non-empty 'kind'" in str(err.value)


def test_deserialize_rejects_check_with_empty_kind_and_no_type():
    payload: dict[str, Any] = {
        "name": "tc-bad-empty-kind",
        "interaction": _minimal_interaction_payload(),
        "checks": [
            {
                "kind": "",
                # explicitly no '__type__'
            }
        ],
    }

    with pytest.raises(ValueError) as err:
        TestCase.deserialize(payload)

    assert "Serialized check must include non-empty 'kind'" in str(err.value)
