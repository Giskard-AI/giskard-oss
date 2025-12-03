from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from functools import partial
from typing import Any, cast, override

from pydantic import Field

from ..core.interaction import BaseInteractionSpec
from ..core.trace import Interaction, Trace
from ..scenarios import execute_code, make_generator
from ..utils.callable import a_callable


@BaseInteractionSpec.register("interaction_spec")
class InteractionSpec[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseInteractionSpec[InputType, OutputType, TraceType]
):
    """Flexible interaction specification supporting static values, callables, and generators.

    This is the default implementation of `BaseInteractionSpec` that provides
    a convenient way to specify interactions with varying levels of dynamism:

    - **Static values**: Direct input/output values
    - **Callables**: Functions that compute inputs/outputs (sync or async)
    - **Generators**: Functions that yield multiple inputs over time

    The `inputs` field can be:
    - A static value of type `InputType`
    - A callable with no arguments that returns `InputType` (or awaitable/generator)
    - A callable that takes the current `Trace` and returns `InputType` (or awaitable/generator)
    - A generator/async generator that yields `InputType` values

    The `outputs` field can be:
    - A static value of type `OutputType`
    - A callable that takes `InputType` and returns `OutputType` (or awaitable)
    - A callable that takes `(InputType, Trace)` and returns `OutputType` (or awaitable)
    - A callable that returns an `Interaction` object directly

    When using generators for inputs, the spec will yield multiple interactions,
    one for each input value produced by the generator. Each interaction receives
    the updated trace (including previous interactions) via the generator protocol.

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
    ```python
    InteractionSpec(
        inputs="Hello",
        outputs="Hi there!",
        metadata={"source": "test"}
    )
    ```

    Callable-based outputs:
    ```python
    InteractionSpec(
        inputs="What is 2+2?",
        outputs=lambda inputs: f"Answer: {eval(inputs)}"
    )
    ```

    Trace-dependent inputs:
    ```python
    InteractionSpec(
        inputs=lambda trace: f"Message #{len(trace.interactions) + 1}",
        outputs=lambda inputs, trace: f"Received: {inputs}"
    )
    ```

    Generator for multiple interactions:
    ```python
    async def input_gen(trace: Trace) -> AsyncGenerator[str, Trace]:
        for i in range(3):
            yield f"Message {i+1}"

    InteractionSpec(
        inputs=input_gen,
        outputs=lambda inputs: f"Echo: {inputs}"
    )
    ```
    """

    inputs: (
        InputType
        | Callable[
            [],
            InputType
            | Awaitable[InputType]
            | Generator[InputType, TraceType, None]
            | AsyncGenerator[InputType, TraceType],
        ]
        | Callable[
            [TraceType],
            InputType
            | Awaitable[InputType]
            | Generator[InputType, TraceType, None]
            | AsyncGenerator[InputType, TraceType],
        ]
    ) = Field(..., description="The inputs of the interaction.")
    outputs: (
        OutputType
        | Callable[
            [InputType],
            OutputType
            | Awaitable[
                OutputType
                | Interaction[InputType, OutputType]
                | Awaitable[Interaction[InputType, OutputType]],
            ],
        ]
        | Callable[
            [InputType, Trace[InputType, OutputType]],
            OutputType
            | Awaitable[
                OutputType
                | Interaction[InputType, OutputType]
                | Awaitable[Interaction[InputType, OutputType]],
            ],
        ]
    ) = Field(..., description="The outputs of the interaction.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the interaction."
    )

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        generator = cast(
            AsyncGenerator[InputType, TraceType],
            await make_generator(self.inputs, trace),
        )
        outputs_callable = a_callable(self.outputs)

        try:
            inputs = await anext(generator)
            while True:
                # Execute user-provided logic to transform inputs into either raw outputs
                # or a fully constructed Interaction instance.
                outputs = await execute_code(partial(outputs_callable, inputs), trace)
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


__all__ = ["InteractionSpec"]
