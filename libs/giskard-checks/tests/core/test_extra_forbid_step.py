"""``Step`` rejects unknown keys instead of silently dropping them.

A misspelled ``checks`` key is the worst variant of the silent-drop bug: the
step validates cleanly with an empty check list, so the scenario runs its
interactions, asserts nothing, and reports success. A suite that silently
tests nothing looks exactly like a suite that passes.
"""

import pytest
from giskard.checks.core.scenario import Scenario, Step
from pydantic import ValidationError


def test_misspelled_checks_key_is_rejected():
    """A typo'd ``checks`` must raise rather than yield a vacuous step."""
    with pytest.raises(ValidationError, match="chekcs"):
        Step.model_validate(
            {"interacts": [], "chekcs": [{"kind": "conformity", "rule": "r"}]}
        )


def test_misspelled_interacts_key_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        Step.model_validate({"interact": [], "checks": []})

    # Named explicitly: ``match="interact"`` would also match the valid
    # ``interacts`` spelling and so would not prove the typo was the cause.
    assert [(err["type"], err["loc"]) for err in exc_info.value.errors()] == [
        ("extra_forbidden", ("interact",))
    ]


def test_misspelled_step_key_is_rejected_through_scenario():
    """The typo must surface when a whole scenario is deserialized."""
    with pytest.raises(ValidationError, match="chekcs"):
        Scenario.model_validate(
            {
                "name": "t",
                "steps": [
                    {
                        "interacts": [],
                        "checks": [],
                        "chekcs": [{"kind": "conformity", "rule": "r"}],
                    }
                ],
            }
        )


def test_valid_step_still_validates():
    step = Step.model_validate(
        {
            "interacts": [{"kind": "interact", "inputs": "hi", "outputs": "yo"}],
            "checks": [],
        }
    )
    assert len(step.interacts) == 1


def test_step_round_trips():
    """``model_dump`` output must still validate under ``extra="forbid"``.

    Asserts revalidation rather than equality: ``Step`` instances holding
    equivalent discriminated children do not compare equal, independently of
    the ``extra`` policy.
    """
    step = Step.model_validate(
        {"interacts": [{"kind": "interact", "inputs": "hi", "outputs": "yo"}]}
    )
    reloaded = Step.model_validate(step.model_dump())
    assert reloaded.model_dump() == step.model_dump()


def test_scenario_still_tolerates_unknown_keys():
    """``Scenario`` stays permissive: remote HF datasets release independently."""
    scenario = Scenario.model_validate(
        {"name": "t", "steps": [], "some_future_upstream_key": "ok"}
    )
    assert scenario.name == "t"
