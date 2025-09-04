from typing import Generic, TypeVar

from giskard_checks.core import Interaction

"""Structured interaction specialization.

`StructuredInteraction[In, Out]` narrows the generic `Interaction` to scenarios
where inputs and outputs are structured (e.g., Pydantic models or dict-like
payloads) rather than free-form text.
"""

In = TypeVar("In")
Out = TypeVar("Out")


class StructuredInteraction(Interaction[In, Out], Generic[In, Out]):
    """Typed interaction carrying structured input and output payloads."""
