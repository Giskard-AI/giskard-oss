from abc import ABC, abstractmethod
from typing import Any

from ..core.context import Context
from ..core.interaction import Interaction
from ..utils.discriminated import Discriminated, discriminated_base


@discriminated_base
class InteractionGenerator(Discriminated, ABC):
    """Base class for interaction generators.

    Interaction generators produce Interaction instances that are used
    by checks. Subclasses should be registered using the
    @InteractionGenerator.register("kind") decorator to enable polymorphic
    serialization and deserialization.

    Examples
    --------
    >>> @InteractionGenerator.register("custom")
    ... class CustomGenerator(InteractionGenerator):
    ...     async def generate(self, context: Context) -> Interaction:
    ...         # Custom generation logic here
    ...         return Interaction(inputs=..., outputs=...)
    """

    @abstractmethod
    async def generate(self, context: Context) -> Interaction[Any, Any]:
        """Generate an interaction based on the provided context.

        Parameters
        ----------
        context : Context
            The context containing previous interactions and metadata.

        Returns
        -------
        Interaction[Any, Any]
            The generated interaction containing inputs and outputs.
        """
        ...
