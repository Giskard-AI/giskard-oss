from typing import Any

from pydantic import Field

from ..core.context import Context
from ..core.interaction_result import InteractionResult
from .base import InteractionGenerator


@InteractionGenerator.register("static")
class Interaction(InteractionGenerator):
    """A static interaction with pre-defined inputs and outputs.

    This is the most common type of interaction generator used for testing.
    It simply wraps pre-defined inputs and outputs into an InteractionResult.

    Attributes
    ----------
    inputs:
        The input payload for the system under test.
    outputs:
        The output produced by the system.
    metadata:
        Optional free-form metadata associated with the interaction.

    Examples
    --------
    >>> interaction = Interaction(
    ...     inputs="Hello, world!",
    ...     outputs="Hi there!"
    ... )
    >>> result = await interaction.generate(Context())
    >>> result.inputs
    'Hello, world!'
    """

    inputs: Any
    outputs: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    async def generate(self, context: Context) -> InteractionResult[Any, Any]:
        """Generate an interaction by returning the pre-defined inputs and outputs.

        Parameters
        ----------
        context : Context
            The context (unused for static interactions).

        Returns
        -------
        InteractionResult[Any, Any]
            The interaction result containing the pre-defined inputs and outputs.
        """
        return InteractionResult(
            inputs=self.inputs,
            outputs=self.outputs,
            metadata=self.metadata,
        )
