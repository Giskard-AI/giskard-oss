from typing import Any, Self

from pydantic import BaseModel, Field, computed_field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.rule import Rule

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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield "Inputs: " + repr(self.inputs)
        yield "Outputs: " + repr(self.outputs)


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
    last : Interaction[InputType, OutputType] | None
        Computed field that returns the last interaction in the trace, or None if empty.
        Equivalent to `interactions[-1]` if interactions exist, None otherwise.
        Available in Python code, Jinja2 prompt templates, and JSONPath expressions.

    Examples
    --------
    ```python
    trace = Trace(interactions=[
        Interaction(inputs="Hello", outputs="Hi there!"),
        Interaction(inputs="How are you?", outputs="I'm doing well, thanks!"),
    ])

    # Access the most recent interaction (preferred in prompt templates and JSONPath)
    last_interaction = trace.last

    # Use in JSONPath expressions
    check = Groundedness(answer_key="trace.last.outputs")

    # Access all outputs
    all_outputs = [interaction.outputs for interaction in trace.interactions]
    ```
    """

    interactions: list[Interaction[InputType, OutputType]] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared Scenario/Trace-level annotations.",
    )

    @computed_field
    @property
    def last(self) -> Interaction[InputType, OutputType] | None:
        """The last interaction in the trace, or None if the trace is empty.

        This computed field is equivalent to `interactions[-1]` when interactions exist.
        It's convenient for use in Python code, Jinja2 prompt templates, and JSONPath
        expressions (e.g., `trace.last.outputs`).

        Examples
        --------
        ```python
        # In Python code
        last_interaction = trace.last

        # In JSONPath expressions
        check = Groundedness(answer_key="trace.last.outputs")

        # In Jinja2 templates
        # {{ trace.last.outputs }}
        ```
        """
        return self.interactions[-1] if self.interactions else None

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
        interaction: (
            Interaction[InputType, OutputType]
            | InteractionGenerator[Interaction[InputType, OutputType], Self]
        ),
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

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        for idx, interaction in enumerate(self.interactions):
            yield Rule(f"Interaction {idx + 1}", style="bold")
            yield from interaction.__rich_console__(console, options)
