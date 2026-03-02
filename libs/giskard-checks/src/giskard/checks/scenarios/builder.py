from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field

from ..core import Interact, Trace
from ..core.check import Check
from ..core.input_generator import InputGenerator
from ..core.interaction import InteractionSpec
from ..core.result import ScenarioResult
from ..core.scenario import Scenario
from ..core.types import GeneratorType, ProviderType


class ScenarioBuilder[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """Builder for creating scenarios with a fluent API.

    This builder allows constructing scenarios step-by-step, providing a more
    user-friendly API than directly instantiating `Scenario` with a flat sequence.

    Examples
    --------
    ```python
    # Build the scenario
    scenario = (
        scenario("my_scenario")
        .interact("Hello", "Hi")
        .check(StringMatching(keyword="Hi", text_key="trace.last.outputs"))
        .interact("How are you?", "Good")
        .check(Equals(expected="Good", key="trace.last.outputs"))
        .build()
    )

    # Or run directly without building
    result = await (
        scenario("my_scenario")
        .interact("Hello", "Hi")
        .check(StringMatching(keyword="Hi", text_key="trace.last.outputs"))
        .run()
    )
    ```
    """

    name: str = Field(..., description="Scenario name")
    sequence: list[
        InteractionSpec[InputType, OutputType, TraceType]
        | Check[InputType, OutputType, TraceType]
    ] = Field(default_factory=list, description="Sequential components to execute")
    trace_type: type[TraceType] | None = Field(
        default=None,
        description="Optional custom trace type. If not provided, will be inferred from components.",
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Scenario-level annotations.",
    )

    def interact(
        self,
        inputs: (
            InputGenerator[InputType, TraceType]
            | GeneratorType[[], InputType, None]
            | GeneratorType[[TraceType], InputType, TraceType]
        ),
        outputs: (
            ProviderType[[InputType], OutputType]
            | ProviderType[[InputType, TraceType], OutputType]
        ),
        metadata: dict[str, object] | None = None,
    ) -> Self:
        """Add an interaction to the scenario sequence.

        Creates an `Interact` with the provided inputs and outputs and adds
        it to the scenario sequence. Supports static values, callables, and generators
        just like `Interact`.

        Parameters
        ----------
        inputs : InputType | Callable | Generator | InputGenerator
            The input specification for the interaction. Can be:
            - A static value of type `InputType`
            - A callable with no arguments that returns `InputType` (or awaitable/generator)
            - A callable that takes the current `Trace` and returns `InputType` (or awaitable/generator)
            - A generator/async generator that yields `InputType` values
            - An `InputGenerator` instance
        outputs : OutputType | Callable
            The output specification for the interaction. Can be:
            - A static value of type `OutputType`
            - A callable that takes `InputType` and returns `OutputType` (or awaitable)
            - A callable that takes `(InputType, Trace)` and returns `OutputType` (or awaitable)
            - A callable that returns an `Interaction` object directly
        metadata : dict[str, object] | None
            Optional metadata to attach to the interaction.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        # Static values
        builder = scenario("my_test")
        builder.interact("Hello", "Hi")

        # Callable outputs
        builder.interact("Hello", lambda inputs: f"Echo: {inputs.upper()}")

        # Trace-dependent inputs
        builder.interact(
            lambda trace: f"Message #{len(trace.interactions) + 1}",
            lambda inputs, trace: f"Received: {inputs}"
        )
        ```
        """
        interaction = Interact(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {},
        )
        self.sequence.append(interaction)
        return self

    def check(self, check: Check[InputType, OutputType, TraceType]) -> Self:
        """Add a check to the scenario sequence.

        Adds a check component to the scenario sequence. Checks validate the
        current trace state and stop execution on failure.

        Parameters
        ----------
        check : Check
            The check to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        builder = scenario("my_test")
        builder.check(StringMatching(keyword="Hi", text_key="trace.last.outputs"))
        ```
        """
        self.sequence.append(check)
        return self

    def checks(self, *checks: Check[InputType, OutputType, TraceType]) -> Self:
        """Add multiple checks to the scenario sequence.

        Adds one or more check components to the scenario sequence. Checks validate
        the current trace state and stop execution on failure.

        Parameters
        ----------
        *checks : Check
            One or more checks to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        builder = scenario("my_test")
        builder.checks(check1, check2, check3)
        builder.checks(*list_of_checks)
        ```
        """
        self.sequence.extend(checks)
        return self

    def add_interaction(
        self,
        interaction: InteractionSpec[InputType, OutputType, TraceType],
    ) -> Self:
        """Add a custom InteractionSpec to the scenario sequence.

        Adds a custom `InteractionSpec` subclass instance to the sequence.
        This is useful when using custom interaction generators or when you need
        more complex interaction generation logic.

        Parameters
        ----------
        interaction : InteractionSpec
            The interaction spec to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        custom_interaction = CustomInteraction(...)
        builder = scenario("my_test")
        builder.add_interaction(custom_interaction)
        ```
        """
        self.sequence.append(interaction)
        return self

    def add_interactions(
        self, *interactions: InteractionSpec[InputType, OutputType, TraceType]
    ) -> Self:
        """Add multiple InteractionSpec objects to the scenario sequence.

        Adds one or more custom `InteractionSpec` subclass instances to the
        sequence. This is useful when using custom interaction generators or when
        you need more complex interaction generation logic.

        Parameters
        ----------
        *interactions : InteractionSpec
            One or more interaction specs to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        interactions = [
            CustomInteraction(...),
            AnotherInteraction(...),
        ]
        builder = scenario("my_test")
        builder.add_interactions(*interactions)
        builder.add_interactions(interaction1, interaction2, interaction3)
        ```
        """
        self.sequence.extend(interactions)
        return self

    def append(
        self,
        component: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Append any component to the scenario sequence.

        Generic method to append any valid scenario component (InteractionSpec or Check) to the sequence. This provides maximum
        flexibility when constructing scenarios.

        Parameters
        ----------
        component : InteractionSpec | Check
            The component to append to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        builder = scenario("my_test")
        builder.append(Interact(inputs="Hello", outputs="Hi"))
        builder.append(CustomInteraction(...))
        builder.append(StringMatching(...))
        ```
        """
        self.sequence.append(component)
        return self

    def extend(
        self,
        *components: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Extend the scenario sequence with multiple components of any type.

        Generic method to extend the sequence with multiple valid scenario components
        (InteractionSpec or Check). This is useful when you
        have a mixed list of components to add.

        Parameters
        ----------
        *components : InteractionSpec | Check
            One or more components of any type to extend the scenario with.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        components = [
            Interact(inputs="Hello", outputs="Hi"),
            CustomInteraction(...),
            StringMatching(...),
        ]
        builder = scenario("my_test")
        builder.extend(*components)
        builder.extend(interaction1, spec1, check1, check2)
        ```
        """
        self.sequence.extend(components)
        return self

    def with_annotations(self, annotations: dict[str, Any]) -> Self:
        """Set scenario-level annotations for the builder.

        Annotations provide shared, read-only context available on the Trace
        as `trace.annotations` during scenario execution.
        """
        self.annotations = annotations
        return self

    def build(self) -> Scenario[InputType, OutputType, TraceType]:
        """Build the scenario from the accumulated sequence.

        Returns
        -------
        Scenario
            The constructed scenario ready to execute.
        """
        return Scenario(
            name=self.name,
            sequence=self.sequence,
            trace_type=self.trace_type,
            annotations=self.annotations,
        )

    async def run(
        self, return_exception: bool = False
    ) -> ScenarioResult[InputType, OutputType]:
        """Build and run the scenario.

        This method automatically builds the scenario before running it, so you can
        call `.run()` directly without needing to call `.build()` first.

        Parameters
        ----------
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.

        Returns
        -------
        ScenarioResult
            Results from executing the scenario.

        Examples
        --------
        ```python
        # Shortcut: run() can be called directly
        result = await scenario("my_test").check(StringMatching(...)).run()

        # Or build first, then run
        scenario_obj = scenario("my_test").check(StringMatching(...)).build()
        result = await scenario_obj.run()
        ```
        """
        return await self.build().run(return_exception=return_exception)


def scenario[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    name: str | None = None,
    trace_type: type[TraceType] | None = None,
    annotations: dict[str, Any] | None = None,
) -> ScenarioBuilder[InputType, OutputType, TraceType]:
    """Create a new scenario builder.

    Parameters
    ----------
    name : str | None
        Name for the scenario. If None, defaults to "Unnamed Scenario".
    trace_type : type[TraceType] | None
        Optional custom trace type to use. If not provided, the trace type will be
        inferred from the sequence of components. Useful when using custom trace
        subclasses with additional computed fields or methods.

    Returns
    -------
    ScenarioBuilder
        A new builder instance ready for configuration.

    Examples
    --------
    ```python
    # Basic usage
    builder = scenario("my_test")
    scenario = builder.build()

    # With custom trace type
    builder = scenario("my_test", trace_type=CustomTrace)
    scenario = builder.build()
    ```
    """
    return ScenarioBuilder(
        name=name or "Unnamed Scenario",
        trace_type=trace_type,
        annotations=annotations or {},
    )
