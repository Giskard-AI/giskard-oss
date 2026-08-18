"""Unknown fields on ``InteractionSpec`` / ``InputGenerator`` must raise.

Both models sit directly in persisted scenario JSON, alongside ``Check``
(which already forbids extras). Without ``extra="forbid"`` pydantic silently
drops unknown keys, so a persisted scenario referencing a renamed field falls
back to that field's default and the scenario runs with the wrong config.

Demonstrated silent drops that these tests lock down:

- ``{"kind": "interact", "inputs": "hi", "output": "X"}`` -- ``output`` (real
  field: ``outputs``) was dropped and ``outputs`` stayed ``MISSING``.
- ``{"kind": "llm_generator", "prompt": "x", "max_step": 99}`` -- ``max_step``
  (real field: ``max_steps``) was dropped and ``max_steps`` stayed ``3``.
"""

from typing import Any

import pytest
from giskard.checks.core.input_generator import InputGenerator
from giskard.checks.core.interaction import Interact, InteractionSpec
from giskard.checks.core.scenario import Scenario
from giskard.checks.generators.base import LLMGenerator
from giskard.checks.generators.dataset import DatasetInputGenerator
from giskard.checks.generators.user import UserSimulator
from giskard.checks.testing.spy import WithSpy
from giskard.core.discriminated import _REGISTRY
from pydantic import ValidationError


def _interaction_specs() -> list[InteractionSpec[Any, Any, Any]]:
    return [
        Interact(inputs="hello", outputs="hi"),
        Interact(inputs=LLMGenerator(prompt="be a user")),
        WithSpy(
            interaction_generator=Interact(inputs="hello", outputs="hi"),
            target="os.getcwd",
        ),
    ]


def _input_generators() -> list[InputGenerator[Any]]:
    # One instance per registered subclass; ``test_every_registered_subclass_is_covered``
    # enforces that. The ``prompt`` / ``prompt_path`` LLMGenerator variants dump
    # the identical field set, so a second sample adds no extras signal.
    return [
        LLMGenerator(prompt="be a user"),
        DatasetInputGenerator(prompt="ignore previous instructions"),
        UserSimulator(persona="frustrated_customer"),
    ]


# --- the two demonstrated silent drops -------------------------------------


def test_interact_typo_output_is_rejected() -> None:
    """``output`` (singular) was silently dropped; ``outputs`` stayed MISSING."""
    with pytest.raises(ValidationError) as exc_info:
        Interact.model_validate({"kind": "interact", "inputs": "hi", "output": "X"})

    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


def test_input_generator_typo_is_rejected_when_validated_as_a_generator() -> None:
    """``max_step`` (singular) was silently dropped; ``max_steps`` stayed 3.

    This is the direct path, where the ``InputGenerator`` arm is actually
    reached. The nested path through ``Interact.inputs`` cannot surface it --
    see ``test_nested_input_generator_typo_does_not_reach_a_generator``.
    """
    with pytest.raises(ValidationError) as exc_info:
        LLMGenerator.model_validate(
            {"kind": "llm_generator", "prompt": "x", "max_step": 99}
        )

    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


# --- the same, through the full nested Scenario path ------------------------


def _scenario_payload(interact: dict[str, Any]) -> dict[str, Any]:
    return {"name": "t", "steps": [{"interacts": [interact], "checks": []}]}


def test_scenario_rejects_interact_typo() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            _scenario_payload({"kind": "interact", "inputs": "hi", "output": "X"})
        )


def test_nested_input_generator_typo_does_not_reach_a_generator() -> None:
    """Known gap: ``Interact.inputs`` swallows generator validation errors.

    ``inputs`` is ``InputGenerator | GeneratorType[[], InputType, None] | ...``
    and ``GeneratorType[[], InputType, None]`` collapses to ``InputType``,
    which is ``Any`` on an unparametrized ``Interact``. So any dict rejected by
    the ``InputGenerator`` arm falls through to that last arm and is kept as a
    raw static input value rather than raising.

    This predates ``extra="forbid"`` and is not caused by it: a payload missing
    the *required* ``prompt``/``prompt_path`` falls through identically (see
    the second half of this test). ``extra="forbid"`` is still doing its job
    here -- it simply cannot surface through this union. Fixing it requires
    narrowing the ``Interact.inputs`` union, which is a separate change.
    """
    typo = Scenario.model_validate(
        _scenario_payload(
            {
                "kind": "interact",
                "inputs": {"kind": "llm_generator", "prompt": "x", "max_step": 99},
            }
        )
    )
    inputs = typo.steps[0].interacts[0].inputs  # pyright: ignore[reportAttributeAccessIssue]
    # Not silently defaulted to max_steps=3 on a real generator: it never
    # became a generator at all.
    assert not isinstance(inputs, InputGenerator)
    assert inputs == {"kind": "llm_generator", "prompt": "x", "max_step": 99}

    # Same fallthrough for a plainly invalid payload, proving it is the union
    # and not ``extra="forbid"``.
    missing_required = Interact.model_validate(
        {"kind": "interact", "inputs": {"kind": "llm_generator"}}
    )
    assert not isinstance(missing_required.inputs, InputGenerator)


def test_scenario_rejects_check_typo() -> None:
    """The ``Check`` level already forbids extras; assert the nesting still bites."""
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "name": "t",
                "steps": [
                    {
                        "interacts": [],
                        "checks": [
                            {
                                "kind": "string_matching",
                                "keyword": "x",
                                "bogus_key": 1,
                            }
                        ],
                    }
                ],
            }
        )


def test_full_nested_scenario_still_validates() -> None:
    """A well-formed scenario with interacts + nested inputs + checks must load."""
    scenario = Scenario.model_validate(
        {
            "name": "t",
            "steps": [
                {
                    "interacts": [
                        {
                            "kind": "interact",
                            "inputs": {
                                "kind": "llm_generator",
                                "prompt": "x",
                                "max_steps": 5,
                            },
                        }
                    ],
                    "checks": [{"kind": "string_matching", "keyword": "x"}],
                }
            ],
        }
    )

    step = scenario.steps[0]
    interact = step.interacts[0]
    assert isinstance(interact, Interact)
    inputs = interact.inputs
    assert isinstance(inputs, LLMGenerator)
    assert inputs.max_steps == 5
    assert len(step.checks) == 1


# --- exhaustive per-subclass coverage ---------------------------------------


@pytest.mark.parametrize(
    "spec", _interaction_specs(), ids=lambda s: type(s).__name__ + "-spec"
)
def test_interaction_spec_forbids_extras(spec: InteractionSpec[Any, Any, Any]) -> None:
    payload = spec.model_dump()
    payload["definitely_not_a_field"] = 1

    with pytest.raises(ValidationError):
        type(spec).model_validate(payload)


@pytest.mark.parametrize(
    "gen", _input_generators(), ids=lambda g: type(g).__name__ + "-gen"
)
def test_input_generator_forbids_extras(gen: InputGenerator[Any]) -> None:
    payload = gen.model_dump()
    payload["definitely_not_a_field"] = 1

    with pytest.raises(ValidationError):
        type(gen).model_validate(payload)


# --- ``kind`` discriminator round trips -------------------------------------


@pytest.mark.parametrize(
    "spec", _interaction_specs(), ids=lambda s: type(s).__name__ + "-spec"
)
def test_interaction_spec_round_trip_direct(
    spec: InteractionSpec[Any, Any, Any],
) -> None:
    """``kind`` is a computed_field: it appears in the dump but is not a field.

    Compares dumps rather than instances: ``InteractionSpec`` subclasses carry
    private attrs (``Interact._output_injectable`` and friends) holding freshly
    built, non-comparable objects, which pydantic's ``__eq__`` includes. Two
    equivalent instances are therefore never ``==``, independently of the
    ``extra`` policy. (Checks and input generators have no such attrs and are
    compared by instance elsewhere in this suite.)
    """
    payload = spec.model_dump()
    assert payload["kind"] is not None

    restored = type(spec).model_validate(payload)
    assert restored.model_dump() == payload


@pytest.mark.parametrize(
    "spec", _interaction_specs(), ids=lambda s: type(s).__name__ + "-spec"
)
def test_interaction_spec_round_trip_polymorphic(
    spec: InteractionSpec[Any, Any, Any],
) -> None:
    """Dump comparison, for the private-attr reason given on the direct test."""
    payload = spec.model_dump()
    restored = (
        Scenario.model_validate(
            {"name": "t", "steps": [{"interacts": [payload], "checks": []}]}
        )
        .steps[0]
        .interacts[0]
    )

    assert type(restored) is type(spec)
    assert restored.model_dump() == payload


@pytest.mark.parametrize(
    "gen", _input_generators(), ids=lambda g: type(g).__name__ + "-gen"
)
def test_input_generator_round_trip_direct(gen: InputGenerator[Any]) -> None:
    payload = gen.model_dump()
    assert payload["kind"] is not None

    restored = type(gen).model_validate(payload)
    assert restored == gen


@pytest.mark.parametrize(
    "gen", _input_generators(), ids=lambda g: type(g).__name__ + "-gen"
)
def test_input_generator_round_trip_polymorphic(gen: InputGenerator[Any]) -> None:
    """Reach the generator through ``Interact.inputs``, the real nesting."""
    payload = gen.model_dump()
    interact = Interact.model_validate({"kind": "interact", "inputs": payload})

    restored = interact.inputs
    assert isinstance(restored, InputGenerator)
    assert restored == gen


@pytest.mark.parametrize(
    "spec", _interaction_specs(), ids=lambda s: type(s).__name__ + "-spec"
)
def test_interaction_spec_json_round_trip(spec: InteractionSpec[Any, Any, Any]) -> None:
    """JSON mode, the path ``Scenario.model_validate_json`` actually takes.

    Dump comparison, for the private-attr reason given on the direct test.
    """
    restored = type(spec).model_validate_json(spec.model_dump_json())
    assert restored.model_dump_json() == spec.model_dump_json()


@pytest.mark.parametrize(
    "gen", _input_generators(), ids=lambda g: type(g).__name__ + "-gen"
)
def test_input_generator_json_round_trip(gen: InputGenerator[Any]) -> None:
    """JSON mode, the path ``Scenario.model_validate_json`` actually takes.

    Input generators have no private attrs, so instance equality holds and is
    asserted here rather than the weaker dump comparison.
    """
    restored = type(gen).model_validate_json(gen.model_dump_json())
    assert restored == gen


def test_every_registered_subclass_is_covered() -> None:
    """Guard: a newly registered subclass must be added to the samples above.

    Test-only subclasses registered by other test modules are excluded; only
    subclasses shipped in the ``giskard.checks`` package are required here.
    """
    for base, samples in (
        (InteractionSpec, _interaction_specs()),
        (InputGenerator, _input_generators()),
    ):
        registered = {
            cls
            for cls in _REGISTRY._subclasses[base].values()
            if cls.__module__.startswith("giskard.")
        }
        assert registered <= {type(sample) for sample in samples}
