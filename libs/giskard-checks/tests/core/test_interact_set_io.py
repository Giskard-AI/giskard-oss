"""Interact input/output providers stay consistent with the live fields.

The runtime resolves an interact's inputs/outputs through private providers.
Those providers are derived lazily from the current fields, so they cannot
desync from the fields no matter how the fields were set — in particular
``model_copy(update=...)``, which skips validation. These tests pin that via an
actual run so the footgun cannot silently regress.
"""

from typing import Any

from giskard.checks import Interact, Scenario, Suite


def _echo(inputs: str) -> str:
    return f"echo: {inputs}"


async def _input_sent(scenario: Scenario[Any, Any, Any]) -> str:
    result = await Suite(name="t").append(scenario).run(target=_echo)
    return result.results[0].final_trace.interactions[-1].inputs


async def test_model_copy_update_inputs_is_used_at_runtime():
    # Regression: model_copy skips validation; a cached provider would keep the
    # original input. The lazy provider must reflect the updated field.
    interact = Interact(inputs="original").model_copy(update={"inputs": "replaced"})
    scenario = Scenario(name="s", steps=[{"interacts": [interact], "checks": []}])

    assert interact.inputs == "replaced"
    assert await _input_sent(scenario) == "replaced"


async def test_direct_assignment_to_inputs_is_used_at_runtime():
    interact = Interact(inputs="original")
    interact.inputs = "replaced"
    scenario = Scenario(name="s", steps=[{"interacts": [interact], "checks": []}])

    assert await _input_sent(scenario) == "replaced"
