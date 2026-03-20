from collections.abc import AsyncGenerator
from typing import Any, cast, override

from giskard.checks.utils.injectable import ValueGenerator, ValueProvider
from giskard.core.utils import NOT_PROVIDED, NotProvided
from pydantic import Field, PrivateAttr, model_validator

from ..input_generator import InputGenerator
from ..types import GeneratorType, ProviderType
from .base import InteractionSpec
from .interaction import Interaction
from .trace import Trace


@InteractionSpec.register("interact")
class Interact[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InteractionSpec[InputType, OutputType, TraceType]
):
    """Defines how to interact with a system.

    `Interact` is an interaction specification representing a logical exchange
    with a system (steps in a workflow, turns in a chat, etc.).

    It describes *how* to generate one or more `Interaction` objects. At runtime,
    its `inputs` and `outputs` specifications are resolved by `Interact.generate`
    to produce a sequence of immutable `Interaction` realizations.

    In the simplest case, `Interact` is a static input and output pair:

        Interact(inputs="Hello", outputs="Hi")

    For dynamic interactions, you can use a callable or a generator:

        Interact(
            inputs=lambda trace: f"Here's a random number: {random.randint(1, 100)}",
            outputs=lambda inputs: f"I received the number {inputs}"
        )

    At test time, the callables will be invoked and produce a realization of
    `Interaction` such as:

        Interaction(inputs="Here's a random number: 42", outputs="I received the number 42")

    If you use a generator, a sequence of `Interaction` realizations will be
    produced, until exhaustion of the generator:

        async def input_generator(trace: Trace) -> AsyncGenerator[str, Trace]:
            for i in range(3):
                yield f"Message {i+1}"

        interact = Interact(inputs=input_generator, outputs=lambda inputs: f"Received: {inputs}")

    At test time, this will produce a sequence of 3 interactions:

        Interaction(inputs="Message 1", outputs="Received: Message 1")
        Interaction(inputs="Message 2", outputs="Received: Message 2")
        Interaction(inputs="Message 3", outputs="Received: Message 3")

    Both `inputs` and `outputs` support static and dynamic forms.

    The `inputs` field can be:
    - A static value
    - A callable with no arguments
    - A callable that takes the current `Trace`
    - A generator/async generator

    The `outputs` field can be:
    - A static value
    - A callable that takes `InputType` arguments
    - A callable that takes `(InputType, Trace)` arguments
    - A callable that returns an `Interaction` object directly

    Awaitable callables will be awaited before being used.

    Attributes
    ----------
    inputs : InputType | Callable[..., InputType | Awaitable[InputType] | Generator | AsyncGenerator]
        Input specification. Can be a static value, callable, or generator.
        Callables can take no arguments or the current `Trace` as an argument.
        Generators yield multiple inputs and receive updated traces via `asend()`.
    outputs : OutputType | Callable[..., OutputType | Awaitable[OutputType | Interaction]]
        Output specification. Can be a static value or callable.
        Callables receive the current `InputType` and optionally the current `Trace`.
        Can return an `Interaction` object directly to override default metadata.
    metadata : dict[str, Any]
        Default metadata to attach to interactions. Can be overridden if `outputs`
        returns an `Interaction` object directly.

    Examples
    --------
    Static inputs and outputs:

    >>> Interact(
    ...     inputs="Hello",
    ...     outputs="Hi there!",
    ...     metadata={"source": "test"}
    ... )
    Interact(inputs='Hello', outputs='Hi there!', metadata=...)

    Callable-based outputs:

    >>> Interact(
    ...     inputs="What is 2+2?",
    ...     outputs=lambda inputs: f"Answer: {eval(inputs)}"
    ... )
    Interact(inputs='What is 2+2?', outputs=<function <lambda> at 0x...>, metadata=...)

    Trace-dependent inputs:

    >>> Interact(
    ...     inputs=lambda trace: f"Message #{len(trace.interactions) + 1}",
    ...     outputs=lambda inputs, trace: f"Received: {inputs}"
    ... )
    Interact(inputs=<function <lambda> at 0x...>, outputs=<function <lambda> at 0x...>, metadata=...)

    Generator for multiple interactions:

    >>> async def input_gen(trace: Trace) -> AsyncGenerator[str, Trace]:
    ...     for i in range(3):
    ...         yield f"Message {i+1}"
    ...
    >>> Interact(
    ...     inputs=input_gen,
    ...     outputs=lambda inputs: f"Echo: {inputs}"
    ... )
    Interact(inputs=<function input_gen at 0x...>, outputs=<function <lambda> at 0x...>, metadata=...)
    """

    inputs: (
        InputGenerator[InputType, TraceType]
        | GeneratorType[[], InputType, None]
        | GeneratorType[[TraceType], InputType, TraceType]
    ) = Field(..., description="The inputs of the interaction.")
    outputs: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, TraceType], OutputType]
        | NotProvided
    ) = Field(default=NOT_PROVIDED, description="The outputs of the interaction.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the interaction."
    )

    _input_value_generator_provider: ValueGenerator[..., InputType, TraceType] = (
        PrivateAttr()
    )
    _output_injectable: ValueProvider[..., OutputType] = PrivateAttr()

    def _validate_inputs(self) -> None:
        try:
            self._input_value_generator_provider = cast(
                ValueGenerator[[TraceType], InputType, TraceType],
                ValueGenerator(self.inputs, {"trace"}),
            )
        except ValueError as e:
            raise ValueError(f"Error getting injection settings for inputs: {e}") from e

    def _validate_outputs(self) -> None:
        try:
            if not isinstance(self.outputs, NotProvided):
                self._output_injectable = ValueProvider(
                    self.outputs, {"inputs", "trace"}
                )
        except ValueError as e:
            raise ValueError(
                f"Error getting injection settings for outputs: {e}"
            ) from e

    @model_validator(mode="after")
    def _validate_injection_mappings(
        self,
    ) -> "Interact[InputType, OutputType, TraceType]":
        self._validate_inputs()
        self._validate_outputs()

        return self

    def set_outputs(
        self,
        outputs: (
            ProviderType[[InputType], OutputType]
            | ProviderType[[InputType, TraceType], OutputType]
            | NotProvided
        ),
    ) -> "Interact[InputType, OutputType, TraceType]":
        """Update the outputs of the interact and recompute the injection mappings. Returns self for chaining."""
        self.outputs = outputs
        self._validate_outputs()

        return self

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        generator = await self._input_value_generator_provider(trace=trace)

        try:
            inputs = await anext(generator)
            while True:
                if isinstance(self.outputs, NotProvided):
                    raise ValueError(
                        "Interaction outputs are not provided and no target was bound."
                    )
                # Execute user-provided logic to transform inputs into either raw outputs
                # or a fully constructed Interaction instance.
                outputs = await self._output_injectable(inputs=inputs, trace=trace)
                # Yield the interaction back to the caller and wait for an updated trace
                # that captures the evaluation of this iteration.
                trace = yield self._get_interaction(
                    inputs,
                    cast(OutputType | Interaction[InputType, OutputType], outputs),
                )
                # Feed the updated trace to the input generator to produce the next inputs.
                inputs = await generator.asend(trace)
        except StopAsyncIteration:
            return
        finally:
            await generator.aclose()

    def _get_interaction(
        self,
        inputs: InputType,
        outputs: OutputType | Interaction[InputType, OutputType],
    ) -> Interaction[InputType, OutputType]:
        return (
            outputs
            if isinstance(outputs, Interaction)
            else Interaction(inputs=inputs, outputs=outputs, metadata=self.metadata)
        )
