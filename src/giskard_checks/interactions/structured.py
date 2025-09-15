from typing import ClassVar, Generic, TypeVar

from giskard_checks.core import Interaction

"""Structured interaction specialization.

`StructuredInteraction[In, Out]` narrows the generic `Interaction` to scenarios
where inputs and outputs are structured (e.g., Pydantic models or dict-like
payloads) rather than free-form text.
"""

In = TypeVar("In")
Out = TypeVar("Out")


class StructuredInteraction(Interaction[In, Out], Generic[In, Out]):
    """Typed interaction carrying structured input and output payloads.

    This specialization of the base `Interaction` class is designed for scenarios
    where both input and output are structured data types (e.g., Pydantic models,
    dictionaries, or other typed objects) rather than free-form text.

    It provides the same interface as the base `Interaction` class but with
    stronger typing guarantees for the input and output fields.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from giskard_checks.interactions import StructuredInteraction
    >>>
    >>> class Input(BaseModel):
    ...     text: str
    >>>
    >>> class Output(BaseModel):
    ...     sentiment: str
    ...     score: float
    >>>
    >>> interaction = StructuredInteraction[Input, Output](
    ...     input=Input(text="Hello world"),
    ...     output=Output(sentiment="positive", score=0.8)
    ... )
    """

    KIND: ClassVar[str | None] = "structured"
