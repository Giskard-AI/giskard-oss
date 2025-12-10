from typing import Any, Self

from pydantic import BaseModel, Field

from .protocols import InteractionGenerator


class Interaction[InputType, OutputType](BaseModel, frozen=True):
    """A single interaction between inputs and outputs.

    An interaction represents one exchange in a conversation or workflow,
    capturing the inputs provided, the outputs produced, and optional metadata.

    Attributes
    ----------
    inputs : InputType
        The input values for this interaction (e.g., user message, API request).
    outputs : OutputType
        The output values produced in response (e.g., assistant reply, API response).
    metadata : dict[str, Any]
        Optional metadata associated with this interaction. Can include timing
        information, tool calls, intermediate states, or any other relevant data.

    Examples
    --------
    ```python
    interaction = Interaction(
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris.",
        metadata={"model": "gpt-4", "tokens": 15}
    )
    ```
    """

    inputs: InputType
    outputs: OutputType
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace[InputType, OutputType](BaseModel, frozen=True):
    """Immutable history of interactions in a scenario.

    A trace accumulates all interactions that have occurred during scenario
    execution. It is passed to checks for validation and to interaction specs
    for generating subsequent interactions.

    The trace is immutable (frozen=True), ensuring that checks and specs cannot
    accidentally modify the history. New interactions are added by creating
    new trace instances.

    Attributes
    ----------
    interactions : list[Interaction[InputType, OutputType]]
        Ordered list of all interactions that have occurred. The most recent
        interaction is at `interactions[-1]`.

    Examples
    --------
    ```python
    trace = Trace(interactions=[
        Interaction(inputs="Hello", outputs="Hi there!"),
        Interaction(inputs="How are you?", outputs="I'm doing well, thanks!"),
    ])

    # Access the most recent interaction
    last_interaction = trace.interactions[-1]

    # Access all outputs
    all_outputs = [interaction.outputs for interaction in trace.interactions]
    ```
    """

    interactions: list[Interaction[InputType, OutputType]] = Field(default_factory=list)

    @classmethod
    async def from_interactions(
        cls,
        *interactions: Interaction[InputType, OutputType]
        | InteractionGenerator[Interaction[InputType, OutputType], Self],
    ) -> Self:
        return await cls().with_interactions(*interactions)

    async def with_interactions(
        self,
        *interactions: Interaction[InputType, OutputType]
        | InteractionGenerator[Interaction[InputType, OutputType], Self],
    ) -> Self:
        trace = self

        for interaction in interactions:
            trace = await trace.with_interaction(interaction)

        return trace

    async def with_interaction(
        self,
        interaction: Interaction[InputType, OutputType]
        | InteractionGenerator[Interaction[InputType, OutputType], Self],
    ) -> Self:
        if isinstance(interaction, Interaction):
            return self.model_copy(
                update={"interactions": self.interactions + [interaction]}
            )

        trace = self
        generator = None
        try:
            generator = interaction.generate(self)
            trace = await self.with_interaction(await anext(generator))
            while True:
                trace = await trace.with_interaction(await generator.asend(trace))
        except StopAsyncIteration:
            return trace
        finally:
            if generator is not None:
                await generator.aclose()

    # TODO def steps() -> list[list[Interaction[InputType, OutputType]]]: # Index based
