from collections.abc import Sequence
from typing import Any, Generic, Self

from giskard.llm.types import (
    ChatMessage,
    ChatMessageParam,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, Field, TypeAdapter, computed_field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.rule import Rule
from typing_extensions import TypeVar

from ..protocols import InteractionGenerator
from .interaction import ChatInteraction, Interaction

InteractionType = TypeVar(
    "InteractionType",
    bound=Interaction[Any, Any],
    default=Interaction[Any, Any],
)


class Trace(BaseModel, Generic[InteractionType], frozen=True):
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
    >>> trace = Trace[Interaction[str, str]](interactions=[
    ...    Interaction(inputs="Hello", outputs="Hi there!"),
    ...    Interaction(inputs="How are you?", outputs="I'm doing well, thanks!"),
    ... ])

    Access the most recent interaction in a trace:
    >>> last_interaction = trace.last
    >>> last_interaction
    Interaction[str, str](inputs='How are you?', ...)

    Use in JSONPath expressions:
    >>> from giskard.checks import Groundedness
    >>> check = Groundedness(answer_key="trace.last.outputs")

    Access all outputs:
    >>> all_outputs = [interaction.outputs for interaction in trace.interactions]
    >>> all_outputs
    ['Hi there!', "I'm doing well, thanks!"]
    """

    interactions: list[InteractionType] = Field(default_factory=list)

    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared Scenario/Trace-level annotations.",
    )

    @computed_field
    @property
    def last(self) -> InteractionType | None:
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
        *interactions: InteractionType | InteractionGenerator[InteractionType, Self],
    ) -> Self:
        return await cls().with_interactions(*interactions)

    async def with_interactions(
        self,
        *interactions: InteractionType | InteractionGenerator[InteractionType, Self],
    ) -> Self:
        trace = self

        for interaction in interactions:
            trace = await trace.with_interaction(interaction)

        return trace

    async def with_interaction(
        self,
        interaction: (InteractionType | InteractionGenerator[InteractionType, Self]),
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


_INPUT_MESSAGE_TYPE = SystemMessage | DeveloperMessage | UserMessage
_CHAT_MESSAGES_TYPE_ADAPTER = TypeAdapter(list[ChatMessage])


class ChatTrace(Trace[ChatInteraction], frozen=True):
    """A trace for a Giskard LLM."""

    @property
    def messages(self) -> list[ChatMessage]:
        return [
            message
            for interaction in self.interactions
            for message in interaction.inputs + interaction.outputs
        ]

    @classmethod
    def from_messages(cls, messages: Sequence[ChatMessage | ChatMessageParam]) -> Self:
        interactions = []
        inputs = []
        outputs = []

        messages = _CHAT_MESSAGES_TYPE_ADAPTER.validate_python(messages)
        for message in messages:
            if isinstance(message, _INPUT_MESSAGE_TYPE):
                if outputs:
                    interactions.append(ChatInteraction(inputs=inputs, outputs=outputs))
                    inputs = []
                    outputs = []
                inputs.append(message)
            else:
                outputs.append(message)

        if inputs or outputs:
            interactions.append(ChatInteraction(inputs=inputs, outputs=outputs))

        return cls(interactions=interactions)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if not self.interactions:
            yield "[dim italic]No messages yet[/dim italic]"
            return

        for interaction in self.interactions:
            yield from interaction.__rich_console__(console, options)
