"""Unknown fields on a Check must raise instead of being silently dropped.

Check subclasses are pydantic models. Without ``extra="forbid"`` pydantic
silently ignores unknown keys, so a saved suite that references a renamed
field falls back to the field default and the check runs GREEN while
evaluating the wrong value (see issue: silent wrong answers on persisted
suites).
"""

from typing import Any

import pytest
from giskard.checks.builtin.comparison import Equals
from giskard.checks.builtin.composition import AllOf
from giskard.checks.builtin.json_valid import JsonValid
from giskard.checks.builtin.text_matching import StringMatching
from giskard.checks.core.check import Check
from giskard.checks.judges.conformity import Conformity
from pydantic import ValidationError


def _make_checks() -> list[Check[Any, Any, Any]]:
    """A representative sample: plain, aliased, judge, composite."""
    inner: list[Check[Any, Any, Any]] = [
        StringMatching(keyword="hello"),
        JsonValid(schema={"type": "object"}),
        Conformity(rule="Be polite."),
    ]
    checks: list[Check[Any, Any, Any]] = [
        StringMatching(keyword="hello"),
        Equals(expected_value=5, target_key="trace.last.outputs"),
        # Aliased field populated; the empty-vs-populated distinction is
        # covered by ``test_json_valid_accepts_both_alias_and_field_name``.
        JsonValid(schema={"type": "object"}),
        Conformity(rule="Be polite."),
        AllOf(checks=inner),
    ]
    return checks


def test_stale_renamed_field_fails_loudly_instead_of_defaulting() -> None:
    """The regression scenario: a saved suite carrying an old field name.

    Previously pydantic dropped the unknown key and ``text_key`` fell back to
    its default ``trace.last.outputs``, so the check ran green against the
    wrong value. It must now error, and specifically with ``extra_forbidden``
    rather than for some incidental reason.
    """
    stale_payload = {
        "kind": "string_matching",
        "keyword": "x",
        # 'text_key' was renamed at some point; the persisted suite still
        # carries the old name.
        "old_text_key": "trace.last.metadata.summary",
    }

    with pytest.raises(ValidationError) as exc_info:
        StringMatching.model_validate(stale_payload)

    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


def test_unknown_field_raises_through_discriminated_registry() -> None:
    """Deserialization via the discriminated union must forbid extras too."""
    with pytest.raises(ValidationError):
        AllOf.model_validate(
            {
                "checks": [
                    {
                        "kind": "string_matching",
                        "keyword": "x",
                        "bogus_key": "trace.last.foo",
                    }
                ]
            }
        )


@pytest.mark.parametrize("check", _make_checks(), ids=lambda c: type(c).__name__)
def test_unknown_field_raises_for_every_sample(check: Check[Any, Any, Any]) -> None:
    payload = check.model_dump()
    payload["definitely_not_a_field"] = 1

    with pytest.raises(ValidationError):
        type(check).model_validate(payload)


@pytest.mark.parametrize("check", _make_checks(), ids=lambda c: type(c).__name__)
def test_model_dump_round_trip(check: Check[Any, Any, Any]) -> None:
    """``model_dump()`` output must still validate back.

    ``kind`` is a ``computed_field``, so it appears in the dump but is not a
    model field. With ``extra="forbid"`` it would be rejected unless it is
    explicitly tolerated.
    """
    payload = check.model_dump()
    assert payload["kind"] is not None

    restored = type(check).model_validate(payload)
    assert restored == check


@pytest.mark.parametrize("check", _make_checks(), ids=lambda c: type(c).__name__)
def test_model_dump_json_round_trip(check: Check[Any, Any, Any]) -> None:
    """The JSON path is how persisted suites are actually loaded.

    ``Scenario.model_validate_json`` is the production entry point (see
    ``giskard.scan.generators.base``), so JSON mode is exercised separately
    from the python-mode dump above rather than assumed equivalent.
    """
    restored = type(check).model_validate_json(check.model_dump_json())
    assert restored == check


def test_json_valid_accepts_both_alias_and_field_name() -> None:
    """``expected_schema`` has ``alias='schema'`` and ``populate_by_name=True``.

    ``extra="forbid"`` must accept both spellings of the aliased field while
    still rejecting a genuinely unknown key alongside them.
    """
    schema = {"type": "object"}

    by_alias = JsonValid.model_validate({"schema": schema})
    by_name = JsonValid.model_validate({"expected_schema": schema})

    assert by_alias.expected_schema == schema
    assert by_name.expected_schema == schema
    # serialize_by_alias=True means the dump uses the alias.
    assert by_alias.model_dump()["schema"] == schema

    with pytest.raises(ValidationError):
        JsonValid.model_validate({"schema": schema, "bogus": 1})
