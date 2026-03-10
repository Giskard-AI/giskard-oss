from collections.abc import Sequence
from typing import Any

from giskard.core.utils import NOT_PROVIDED, NotProvided
from pydantic import BaseModel, Field

from .check import Check
from .interaction import InteractionSpec, Trace
from .result import ScenarioResult
from .types import ProviderType


class Scenario[InputType, OutputType, TraceType: Trace](BaseModel, frozen=True):  # pyright: ignore[reportMissingTypeArgument]
    """A scenario composed of an ordered sequence of components InteractionSpecs
    or Checks with a shared trace.

    A scenario executes components sequentially, maintaining a shared trace that
    accumulates all interactions. Execution stops immediately if any check fails.

    Components are processed in order:
    - **InteractionSpec** components: Add interactions to the trace
    - **Check** components: Validate the current trace state

    Attributes
    ----------
    name : str
        Scenario identifier.
    sequence : Sequence[InteractionSpec | Check]
        Sequential steps to execute. Each component can be an InteractionSpec or
        a Check (which validates the current trace).
    trace_type : type[TraceType] | None
        Optional custom trace type to use. If not provided, the trace type will be
        inferred from the sequence of components. Useful when using custom trace
        subclasses with additional computed fields or methods.

    Examples
    --------
    **Recommended**: Use the fluent API:

        from giskard.checks import scenario, Equals
        result = await (
            scenario("multi_step_test")
            .interact("Hello", lambda inputs: "Hi")
            .check(Equals(expected="Hi", key="trace.last.outputs"))
            .run()
        )

    **Advanced**: Direct instantiation:

        from giskard.checks import Scenario, Interact, Equals

        scenario = Scenario(
            name="multi_step_test",
            sequence=[
                Interact(inputs="Hello", outputs="Hi"),
                Equals(expected="Hi", key="trace.last.outputs"),
            ],
        )
        result = await scenario.run()
    """

    name: str = Field(..., description="Scenario name")
    sequence: Sequence[
        InteractionSpec[InputType, OutputType, TraceType]
        | Check[InputType, OutputType, TraceType]
    ] = Field(..., description="Sequential components to execute")
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
