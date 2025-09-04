# interactlab/core/interaction.py
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

"""Generic interaction model.

An `Interaction` represents an input, an optional output (e.g., a model
response), and optional metadata captured during evaluation. Concrete
specializations can refine the input/output types as needed.
"""

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(BaseModel, Generic[InputT, OutputT]):
    """Container for a single interaction under test.

    Attributes
    ----------
    input:
        The input payload for the system under test.
    output:
        Optional output produced by the system.
    metadata:
        Optional free-form metadata associated with the interaction.
    """

    input: InputT
    output: OutputT | None = None
    metadata: dict[str, Any] | None = None
