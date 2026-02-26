from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field

from ..core.check import Check
from ..core.input_generator import InputGenerator
from ..core.interaction import BaseInteractionSpec
from ..core.result import ScenarioResult
from ..core.scenario import Scenario
from ..core.trace import Interaction, Trace
from ..core.types import GeneratorType, ProviderType
from ..interaction import InteractionSpec


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
        Interaction[InputType, OutputType]
        | BaseInteractionSpec[InputType, OutputType, TraceType]
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

        Creates an `InteractionSpec` with the provided inputs and outputs and adds
        it to the scenario sequence. Supports static values, callables, and generators
        just like `InteractionSpec`.

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
        interaction_spec = InteractionSpec(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {},
        )
        self.sequence.append(interaction_spec)
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

    def add_interaction(self, interaction: Interaction[InputType, OutputType]) -> Self:
        """Add a direct Interaction object to the scenario sequence.

        Adds a pre-constructed `Interaction` object directly to the sequence.
        This is useful when you need more control over the interaction metadata
        or when working with interactions that were created elsewhere.

        Parameters
        ----------
        interaction : Interaction
            The interaction object to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        interaction = Interaction(
            inputs="Hello",
            outputs="Hi",
            metadata={"source": "test"}
        )
        builder = scenario("my_test")
        builder.add_interaction(interaction)
        ```
        """
        self.sequence.append(interaction)
        return self

    def add_interactions(
        self, *interactions: Interaction[InputType, OutputType]
    ) -> Self:
        """Add multiple Interaction objects to the scenario sequence.

        Adds one or more pre-constructed `Interaction` objects directly to the
        sequence. This is useful when you need more control over the interaction
        metadata or when working with interactions that were created elsewhere.

        Parameters
        ----------
        *interactions : Interaction
            One or more interaction objects to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        interactions = [
            Interaction(inputs="Hello", outputs="Hi"),
            Interaction(inputs="How are you?", outputs="Good"),
        ]
        builder = scenario("my_test")
        builder.add_interactions(*interactions)
        builder.add_interactions(interaction1, interaction2, interaction3)
        ```
        """
        self.sequence.extend(interactions)
        return self

    def add_interaction_spec(
        self,
        spec: BaseInteractionSpec[InputType, OutputType, TraceType],
    ) -> Self:
        """Add a custom InteractionSpec to the scenario sequence.

        Adds a custom `BaseInteractionSpec` subclass instance to the sequence.
        This is useful when using custom interaction generators or when you need
        more complex interaction generation logic.

        Parameters
        ----------
        spec : BaseInteractionSpec
            The interaction spec to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        custom_spec = CustomInteractionSpec(...)
        builder = scenario("my_test")
        builder.add_interaction_spec(custom_spec)
        ```
        """
        self.sequence.append(spec)
        return self

    def add_interaction_specs(
        self, *specs: BaseInteractionSpec[InputType, OutputType, TraceType]
    ) -> Self:
        """Add multiple InteractionSpec objects to the scenario sequence.

        Adds one or more custom `BaseInteractionSpec` subclass instances to the
        sequence. This is useful when using custom interaction generators or when
        you need more complex interaction generation logic.

        Parameters
        ----------
        *specs : BaseInteractionSpec
            One or more interaction specs to add to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        specs = [
            CustomInteractionSpec(...),
            AnotherInteractionSpec(...),
        ]
        builder = scenario("my_test")
        builder.add_interaction_specs(*specs)
        builder.add_interaction_specs(spec1, spec2, spec3)
        ```
        """
        self.sequence.extend(specs)
        return self

    def append(
        self,
        component: (
            Interaction[InputType, OutputType]
            | BaseInteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Append any component to the scenario sequence.

        Generic method to append any valid scenario component (Interaction,
        BaseInteractionSpec, or Check) to the sequence. This provides maximum
        flexibility when constructing scenarios.

        Parameters
        ----------
        component : Interaction | BaseInteractionSpec | Check
            The component to append to the scenario.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        builder = scenario("my_test")
        builder.append(Interaction(inputs="Hello", outputs="Hi"))
        builder.append(CustomInteractionSpec(...))
        builder.append(StringMatching(...))
        ```
        """
        self.sequence.append(component)
        return self

    def extend(
        self,
        *components: (
            Interaction[InputType, OutputType]
            | BaseInteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Extend the scenario sequence with multiple components of any type.

        Generic method to extend the sequence with multiple valid scenario components
        (Interaction, BaseInteractionSpec, or Check). This is useful when you
        have a mixed list of components to add.

        Parameters
        ----------
        *components : Interaction | BaseInteractionSpec | Check
            One or more components of any type to extend the scenario with.

        Returns
        -------
        ScenarioBuilder
            Self for method chaining.

        Examples
        --------
        ```python
        components = [
            Interaction(inputs="Hello", outputs="Hi"),
            CustomInteractionSpec(...),
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
