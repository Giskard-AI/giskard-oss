from abc import ABC, abstractmethod
from typing import Any

from ..core.context import Context
from ..core.interaction_result import InteractionResult
from ..utils.discriminated import Discriminated, discriminated_base


@discriminated_base
class InteractionGenerator(Discriminated, ABC):
    """Base class for interaction generators.

    Interaction generators produce InteractionResult instances that are used
    by checks. Subclasses should be registered using the
    @InteractionGenerator.register("kind") decorator to enable polymorphic
    serialization and deserialization.

    Examples
    --------
    >>> @InteractionGenerator.register("custom")
    ... class CustomGenerator(InteractionGenerator):
    ...     async def generate(self, context: Context) -> InteractionResult:
    ...         # Custom generation logic here
    ...         return InteractionResult(inputs=..., outputs=...)
    """

    @abstractmethod
    async def generate(self, context: Context) -> InteractionResult[Any, Any]:
        """Generate an interaction based on the provided context.

        Parameters
        ----------
        context : Context
            The context containing previous interactions and metadata.

        Returns
        -------
        InteractionResult[Any, Any]
            The generated interaction result containing inputs and outputs.
        """
        ...
