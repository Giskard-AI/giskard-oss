"""Interact step metadata survives a target that returns its own Interaction.

Scan generators attach reference data to the step (the knowledge-base scenarios put
the reference documents in ``metadata["context"]`` and check
``trace.last.metadata.context``). A target that returns an ``Interaction`` to attach
its own metadata (tool calls, latency) must not erase that, or every such check
fails with a missing context key.
"""

from typing import Any

from giskard.checks import Interact, Interaction, Trace


async def _first_interaction(
    interact: Interact[Any, Any, Any],
) -> Interaction[Any, Any]:
    generator = interact.generate(Trace(interactions=[]))
    interaction = await anext(generator)
    await generator.aclose()
    return interaction


async def test_step_metadata_is_kept_when_target_returns_an_interaction():
    interact = Interact(
        inputs="q",
        outputs=lambda inputs: Interaction(
            inputs=inputs, outputs="a", metadata={"tool_calls": []}
        ),
        metadata={"context": "reference"},
    )

    interaction = await _first_interaction(interact)

    assert interaction.metadata == {"context": "reference", "tool_calls": []}


async def test_target_metadata_wins_on_a_key_clash():
    interact = Interact(
        inputs="q",
        outputs=lambda inputs: Interaction(
            inputs=inputs, outputs="a", metadata={"source": "target"}
        ),
        metadata={"source": "step"},
    )

    interaction = await _first_interaction(interact)

    assert interaction.metadata == {"source": "target"}
