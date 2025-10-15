from __future__ import annotations

from typing import Any, Generic, TypeVar

from ..utils.discriminated import Discriminated

"""Generic interaction model.

An `Interaction` represents an input, an optional output (e.g., a model
response), and optional metadata captured during evaluation. Concrete
specializations can refine the input/output types as needed.
"""

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(Discriminated, Generic[InputT, OutputT]):
    """Container for a single interaction under test.

    Subclasses should be registered using the @Interaction.register("kind") decorator
    to enable polymorphic serialization and deserialization.

    Attributes
    ----------
    input:
        The input payload for the system under test.
    output:
        Optional output produced by the system.
    metadata:
        Optional free-form metadata associated with the interaction.
    """

    inputs: InputT
    outputs: OutputT | None = None
    metadata: dict[str, Any] | None = None
