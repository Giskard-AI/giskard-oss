from typing import Any, Self

from giskard.core.utils import NOT_PROVIDED, NotProvided
from pydantic import BaseModel, Field

from .check import Check
from .input_generator import InputGenerator
from .interaction import Interact, InteractionSpec, Trace
from .result import ScenarioResult
from .types import GeneratorType, ProviderType


class Scenario[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """A scenario composed of an ordered sequence of components InteractionSpecs
    or Checks with a shared trace.

    A scenario executes components sequentially, maintaining a shared trace that
    accumulates all interactions. Execution stops immediately if any check fails.

    Components are processed in order:
    - **InteractionSpec** components: Add interactions to the trace
    - **Check** components: Validate the current trace state

    Use the fluent API to build a scenario, then call ``run()``:

        from giskard.checks import Scenario, Equals
        scenario = (
            Scenario("multi_step_test")
            .interact("Hello", lambda inputs: "Hi")
            .check(Equals(expected="Hi", key="trace.last.outputs"))
        )
        result = await scenario.run()

    For advanced usage you can instantiate with a pre-filled sequence:

        from giskard.checks import Scenario, Interact, Equals
        scenario = Scenario(
            name="multi_step_test",
            sequence=[
                Interact(inputs="Hello", outputs="Hi"),
                Equals(expected="Hi", key="trace.last.outputs"),
            ],
        )
        result = await scenario.run()

    Attributes
    ----------
    name : str
        Scenario identifier.
    sequence : list[InteractionSpec | Check]
        Sequential steps to execute. Each component can be an InteractionSpec or
        a Check (which validates the current trace).
    trace_type : type[TraceType] | None
        Optional custom trace type to use. If not provided, the trace type will be
        inferred from the sequence of components. Useful when using custom trace
        subclasses with additional computed fields or methods.
    """

    name: str = Field(
        default="Unnamed Scenario",
        description="Scenario name",
    )
    sequence: list[
        InteractionSpec[InputType, OutputType, TraceType]
        | Check[InputType, OutputType, TraceType]
    ] = Field(default_factory=list, description="Sequential components to execute")
    trace_type: type[TraceType] | None = Field(
        default=None,
        description="Type of trace to use for the scenario. If not provided, the trace type will be inferred from the sequence of components.",
    )
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description="Scenario-level annotations that will be injected in the trace.",
    )
    target: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, TraceType], OutputType]
        | NotProvided
    ) = Field(
        default=NOT_PROVIDED,
        description="Scenario-level target SUT that will be used to replace NOT_PROVIDED outputs.",
    )

    def __init__(
        self,
        name: str | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Initialize a scenario. Name can be passed positionally: Scenario(\"my_name\")."""
        if name is not None:
            kwargs["name"] = name
        super().__init__(**kwargs)

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
            | NotProvided
        ) = NOT_PROVIDED,
        metadata: dict[str, object] | None = None,
    ) -> Self:
        """Add an interaction to the scenario sequence.

        Creates an `Interact` with the provided inputs and outputs and adds
        it to the scenario sequence. Supports static values, callables, and generators
        just like `Interact`.

        Parameters
        ----------
        inputs : InputType | Callable | Generator | InputGenerator
            The input specification for the interaction.
        outputs : OutputType | Callable
            The output specification for the interaction.
        metadata : dict[str, object] | None
            Optional metadata to attach to the interaction.

        Returns
        -------
        Scenario
            Self for method chaining.
        """
        interaction = Interact(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {},
        )
        self.sequence.append(interaction)
        return self

    def check(self, check: Check[InputType, OutputType, TraceType]) -> Self:
        """Add a check to the scenario sequence."""
        self.sequence.append(check)
        return self

    def checks(self, *checks: Check[InputType, OutputType, TraceType]) -> Self:
        """Add multiple checks to the scenario sequence."""
        self.sequence.extend(checks)
        return self

    def add_interaction(
        self,
        interaction: InteractionSpec[InputType, OutputType, TraceType],
    ) -> Self:
        """Add a custom InteractionSpec to the scenario sequence."""
        self.sequence.append(interaction)
        return self

    def add_interactions(
        self, *interactions: InteractionSpec[InputType, OutputType, TraceType]
    ) -> Self:
        """Add multiple InteractionSpec objects to the scenario sequence."""
        self.sequence.extend(interactions)
        return self

    def append(
        self,
        component: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Append any component to the scenario sequence."""
        self.sequence.append(component)
        return self

    def extend(
        self,
        *components: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Extend the scenario sequence with multiple components of any type."""
        self.sequence.extend(components)
        return self

    def with_annotations(self, annotations: dict[str, Any]) -> Self:
        """Set scenario-level annotations.

        Annotations provide shared, read-only context available on the Trace
        as `trace.annotations` during scenario execution.
        """
        self.annotations = annotations
        return self

    def with_target(
        self,
        target: (
            ProviderType[[InputType], OutputType]
            | ProviderType[[InputType, TraceType], OutputType]
        ),
    ) -> Self:
        """Set scenario-level target for the scenario."""
        self.target = target
        return self

    async def run(
        self,
        target: (
            ProviderType[[InputType], OutputType]
            | ProviderType[[InputType, TraceType], OutputType]
            | NotProvided
        ) = NOT_PROVIDED,
        return_exception: bool = False,
    ) -> ScenarioResult[InputType, OutputType]:
        """Execute the scenario components sequentially with shared trace.

        Each component is executed in order:
        - Interaction components update the shared trace
        - Check components validate the current trace and stop execution on failure

        Returns
        -------
        ScenarioResult
            Results from executing the scenario components.
        """
        # Lazy import to avoid circular dependency
        from ..scenarios.runner import get_runner

        runner = get_runner()
        return await runner.run(self, target=target, return_exception=return_exception)
