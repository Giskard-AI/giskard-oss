"""Interact input/output providers stay consistent with the live fields.

The runtime resolves an interact's inputs/outputs through providers derived
from the current fields. Those providers must not desync from the fields no
matter how the fields were set — in particular ``model_copy(update=...)``,
which skips validation. These tests pin that via ``generate()`` so the
footgun cannot silently regress.
"""

from typing import Any

from giskard.checks import Interact, Trace
from giskard.checks.generators.dataset import DatasetInputGenerator


async def _generated_inputs(interact: Interact[Any, Any, Any]) -> Any:
    generator = interact.generate(Trace(interactions=[]))
    record = await anext(generator)
    await generator.aclose()
    return record.inputs


async def test_model_copy_update_inputs_is_used_at_runtime():
    """model_copy skips validation; generate() must still use the updated input."""
    interact = Interact(inputs="original", outputs=lambda inputs: inputs).model_copy(
        update={"inputs": "replaced"}
    )

    assert interact.inputs == "replaced"
    assert await _generated_inputs(interact) == "replaced"


async def test_direct_assignment_to_inputs_is_used_at_runtime():
    interact = Interact(inputs="original", outputs=lambda inputs: inputs)
    interact.inputs = "replaced"

    assert await _generated_inputs(interact) == "replaced"


async def test_model_copy_update_dataset_inputs_is_used_at_runtime():
    """GCG (and similar generators) replace DatasetInputGenerator via model_copy."""
    interact = Interact(
        inputs=DatasetInputGenerator(prompt="RAW_PROMPT"),
        outputs=lambda inputs: inputs,
    ).model_copy(update={"inputs": DatasetInputGenerator(prompt="RAW_PROMPT SUFFIX")})

    assert await _generated_inputs(interact) == "RAW_PROMPT SUFFIX"
