from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .check import Check
from .interaction import BaseInteractionSpec
from .result import ScenarioResult
from .trace import Interaction, Trace


class Scenario[InputType, OutputType, TraceType: Trace](BaseModel, frozen=True):  # pyright: ignore[reportMissingTypeArgument]
    """A scenario composed of an ordered sequence of components (InteractionSpecs
    and Checks) with shared trace.

    A scenario executes components sequentially, maintaining a shared trace that
    accumulates all interactions. Execution stops immediately if any check fails.

    Components are processed in order:
    - **InteractionSpec** components: Generate interactions and add them to the trace
    - **Check** components: Validate the current trace state

    Attributes
    ----------
    name : str
        Scenario identifier.
    sequence : Sequence[ScenarioComponent[InputType, OutputType]]
        Sequential steps to execute. Each component can be either an InteractionSpec
        (which updates the trace) or a Check (which validates the current trace).
    trace_type : type[TraceType] | None
        Optional custom trace type to use. If not provided, the trace type will be
        inferred from the sequence of components. Useful when using custom trace
        subclasses with additional computed fields or methods.

    Examples
    --------
    ```python
    scenario = Scenario(
        name="multi_step_test",
        sequence=[
            InteractionSpec(inputs="Hello", outputs="Hi"),
            Equality(expected="Hi", key="trace.interactions[-1].outputs"),
        ],
    )
    result = await scenario.run()
    ```
    """

    name: str = Field(..., description="Scenario name")
    sequence: Sequence[
        Interaction[InputType, OutputType]
        | BaseInteractionSpec[InputType, OutputType, TraceType]
        | Check[InputType, OutputType, TraceType]
    ] = Field(..., description="Sequential components to execute")
    trace_type: type[TraceType] | None = Field(
        default=None,
        description="Type of trace to use for the scenario. If not provided, the trace type will be inferred from the sequence of components.",
    )

    async def run(
        self, return_exception: bool = False
    ) -> ScenarioResult[InputType, OutputType]:
        """Execute the scenario components sequentially with shared trace.

        Each component is executed in order:
        - InteractionSpec components update the shared trace
        - Check components validate the current trace and stop execution on failure

        Returns
        -------
        ScenarioResult
            Results from executing the scenario components.
        """
        # Lazy import to avoid circular dependency
        from ..scenarios.runner import get_runner

        runner = get_runner()
        return await runner.run(self, return_exception)
