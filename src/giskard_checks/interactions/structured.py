from typing import Generic, TypeVar

from giskard_checks.core import Interaction

In = TypeVar("In")
Out = TypeVar("Out")


class StructuredInteraction(Interaction[In, Out], Generic[In, Out]):
    pass
