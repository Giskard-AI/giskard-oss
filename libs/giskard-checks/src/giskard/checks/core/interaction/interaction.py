from collections.abc import AsyncGenerator
from typing import Any, cast, override

from pydantic import Field, PrivateAttr, model_validator

from ...utils.parameter_injection import ParameterInjectionRequirement
from ...utils.value_provider import (
    ValueGeneratorProvider,
    ValueProvider,
)
from ..input_generator import InputGenerator
from ..types import GeneratorType, ProviderType
from .base import BaseInteraction
from .interaction_record import InteractionRecord
from .trace import Trace

INJECTABLE_TRACE = ParameterInjectionRequirement(
    class_info=Trace,
    optional=True,
)

INJECTABLE_INPUT = ParameterInjectionRequirement(
    class_info=Any,
    optional=True,
)


@BaseInteraction.register("interaction")
class Interaction[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseInteraction[InputType, OutputType, TraceType]
):
    """Represents a single exchange with a system (step in a workflow, turn in a chat, etc.)

    Each interaction consists of:
    - **Inputs**: The input values provided to the system
    - **Outputs**: The output values produced by the system
    - **Metadata**: Optional metadata associated with the interaction

    The Interaction class support both static and dynamic inputs and outputs.
    Dynamic values will be evaluated at runtime to create an immutable
    `InteractionRecord` object which contains the observed concrete values.

    The `inputs` field can be:
    - A static value
    - A callable with no arguments
    - A callable that takes the current `Trace`
    - A generator/async generator

    The `outputs` field can be:
    - A static value
    - A callable that takes `InputType` arguments
    - A callable that takes `(InputType, Trace)` arguments
    - A callable that returns an `InteractionRecord` object directly

    Awaitable callables will be awaited before being used.

    Attributes
    ----------
    inputs : InputType | Callable[..., InputType | Awaitable[InputType] | Generator | AsyncGenerator]
        Input specification. Can be a static value, callable, or generator.
        Callables can take no arguments or the current `Trace` as an argument.
        Generators yield multiple inputs and receive updated traces via `asend()`.
    outputs : OutputType | Callable[..., OutputType | Awaitable[OutputType | InteractionRecord]]
        Output specification. Can be a static value or callable.
        Callables receive the current `InputType` and optionally the current `Trace`.
        Can return an `InteractionRecord` object directly to override default metadata.
    metadata : dict[str, Any]
        Default metadata to attach to interactions. Can be overridden if `outputs`
        returns an `InteractionRecord` object directly.

    Examples
    --------
    Static inputs and outputs:
    ```python
    Interaction(
        inputs="Hello",
        outputs="Hi there!",
        metadata={"source": "test"}
    )
    ```

    Callable-based outputs:
    ```python
    Interaction(
        inputs="What is 2+2?",
        outputs=lambda inputs: f"Answer: {eval(inputs)}"
    )
    ```

    Trace-dependent inputs:
    ```python
    Interaction(
        inputs=lambda trace: f"Message #{len(trace.interactions) + 1}",
        outputs=lambda inputs, trace: f"Received: {inputs}"
    )
    ```

    Generator for multiple interactions:
    ```python
    async def input_gen(trace: Trace) -> AsyncGenerator[str, Trace]:
        for i in range(3):
            yield f"Message {i+1}"

    Interaction(
        inputs=input_gen,
        outputs=lambda inputs: f"Echo: {inputs}"
    )
    ```
    """

    inputs: (
        InputGenerator[InputType, TraceType]
        | GeneratorType[[], InputType, None]
        | GeneratorType[[TraceType], InputType, TraceType]
    ) = Field(..., description="The inputs of the interaction.")
    outputs: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, TraceType], OutputType]
    ) = Field(..., description="The outputs of the interaction.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the interaction."
    )

    _input_value_generator_provider: ValueGeneratorProvider[
        [TraceType], InputType, TraceType
    ] = PrivateAttr()
    _output_value_provider: ValueProvider[[InputType, TraceType], OutputType] = (
        PrivateAttr()
    )

    @model_validator(mode="after")
    def _validate_injection_mappings(
        self,
    ) -> "Interaction[InputType, OutputType, TraceType]":
        try:
            self._input_value_generator_provider = ValueGeneratorProvider.from_mapping(
                self.inputs, INJECTABLE_TRACE
            )
        except ValueError as e:
            raise ValueError(f"Error getting injection settings for inputs: {e}") from e

        try:
            self._output_value_provider = ValueProvider.from_mapping(
                self.outputs, INJECTABLE_INPUT, INJECTABLE_TRACE
            )
        except ValueError as e:
            raise ValueError(
                f"Error getting injection settings for outputs: {e}"
            ) from e

        return self

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[InteractionRecord[InputType, OutputType], TraceType]:
        generator = await self._input_value_generator_provider(trace)

        try:
            inputs = await anext(generator)
            while True:
                # Execute user-provided logic to transform inputs into either raw outputs
                # or a fully constructed Interaction instance.
                outputs = await self._output_value_provider(inputs, trace)
                # Yield the interaction back to the caller and wait for an updated trace
                # that captures the evaluation of this iteration.
                trace = yield self._get_interaction_record(
                    inputs,
                    cast(
                        OutputType | InteractionRecord[InputType, OutputType], outputs
                    ),
                )
                # Feed the updated trace to the input generator to produce the next inputs.
                inputs = await generator.asend(trace)
        except StopAsyncIteration:
            return
        finally:
            await generator.aclose()

    def _get_interaction_record(
        self,
        inputs: InputType,
        outputs: OutputType | InteractionRecord[InputType, OutputType],
    ) -> InteractionRecord[InputType, OutputType]:
        return (
            outputs
            if isinstance(outputs, InteractionRecord)
            else InteractionRecord(
                inputs=inputs, outputs=outputs, metadata=self.metadata
            )
        )
