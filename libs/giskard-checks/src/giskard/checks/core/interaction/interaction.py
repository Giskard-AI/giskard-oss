from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console, ConsoleOptions, RenderResult


class Interaction[InputType, OutputType](BaseModel, frozen=True):
    """An immutable record of a single exchange in a chat or workflow.

    Unlike `InteractionSpec` (which dynamically generates inputs at runtime),
    a `Interaction` is fully static and predictable. It captures the concrete
    inputs, outputs, and optional metadata as they occurred. Used within a
    `Trace` to represent the materialized history of a scenario.

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
    >>> Interaction(
    ...    inputs="What is the capital of France?",
    ...    outputs="The capital of France is Paris.",
    ...    metadata={"model": "gpt-4", "tokens": 15}
    ... )
    Interaction(inputs='What is the capital of France?', outputs='The capital of Franc...)
    """

    inputs: InputType
    outputs: OutputType
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield "Inputs: " + repr(self.inputs)
        yield "Outputs: " + repr(self.outputs)
