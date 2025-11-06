from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

"""Generic interaction model.

An `Interaction` represents the result of generating an interaction,
containing inputs, outputs, and optional metadata. This is an internal data
structure used by checks to access interaction data.
"""

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(BaseModel, Generic[InputT, OutputT]):
    """Container for interaction data used internally by checks.

    This is the result of calling `InteractionGenerator.generate()`. It contains
    the inputs, outputs, and metadata for a single interaction.

    Attributes
    ----------
    inputs:
        The input payload for the system under test.
    outputs:
        Optional output produced by the system.
    metadata:
        Optional free-form metadata associated with the interaction.
    """

    inputs: InputT
    outputs: OutputT | None = None
    metadata: dict[str, Any] | None = None
