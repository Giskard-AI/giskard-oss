from collections.abc import AsyncGenerator

from giskard.core import Discriminated, discriminated_base

from .interaction import Interaction
from .trace import Trace


@discriminated_base
class InteractionSpec[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Discriminated
):
    """Base class for interaction specifications that generate interactions.

    An interaction spec produces one or more `Interaction` objects by yielding
    them through an async generator. Each yielded interaction receives the updated
    trace (including the newly yielded interaction) via `generator.asend()`.

    This allows for multi-turn interactions where subsequent inputs can depend
    on the accumulated trace history.

    Subclasses must implement `generate()` to produce interactions. They should
    be registered using `@InteractionSpec.register("kind")` for polymorphic
    serialization.

    Attributes
    ----------
    InputType : TypeVar
        Type of the input values for interactions
    OutputType : TypeVar
        Type of the output values for interactions
    """

    def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        """Generate interactions from the current trace state.

        This method is called by the scenario runner to produce interactions.
        It yields `Interaction` objects and receives updated traces (including
        the newly yielded interaction) via the async generator protocol.

        Parameters
        ----------
        trace : TraceType
            The current trace state before generating the interaction.

        Yields
        ------
        Interaction[InputType, OutputType]
            An interaction record to add to the trace.

        Receives
        --------
        TraceType
            The updated trace after the yielded interaction was added.
            Use `generator.asend(updated_trace)` to receive this value.

        Examples
        --------
        ```python
        async def generate(self, trace: TraceType) -> AsyncGenerator[Interaction, TraceType]:
            record = Interaction(inputs="hello", outputs="hi")
            updated_trace = yield record

            next_input = f"Previous had {len(updated_trace.interactions)} interactions"
            record = Interaction(inputs=next_input, outputs="response")
            yield record
        ```
        """
        raise NotImplementedError
