from typing import Any, Self

from giskard.core.utils import NOT_PROVIDED, NotProvided
from pydantic import BaseModel, Field

from .check import Check
from .input_generator import InputGenerator
from .interaction import Interact, InteractionSpec, Trace
from .result import ScenarioResult
from .types import GeneratorType, ProviderType


class Step[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """A scenario step: a sequence of interactions followed by checks.

    Each step corresponds to one TestCase at runtime: interactions are applied
    to the trace, then checks validate the resulting trace state.
    """

    interacts: list[InteractionSpec[InputType, OutputType, TraceType]] = Field(
        default_factory=list,
        description="Interaction specs to execute in this step.",
    )
    checks: list[Check[InputType, OutputType, TraceType]] = Field(
        default_factory=list,
        description="Checks to run after the interactions.",
    )


class Scenario[InputType, OutputType, TraceType: Trace](BaseModel):  # pyright: ignore[reportMissingTypeArgument]
    """A scenario composed of steps, each containing interacts and checks.

    A scenario executes steps sequentially, maintaining a shared trace that
    accumulates all interactions. Execution stops immediately if any check fails.

    Each step groups:
    - **Interacts** (InteractionSpec): Add interactions to the trace
    - **Checks**: Validate the current trace state

    Use the fluent API to build a scenario, then call ``run()``:

        from giskard.checks import Scenario, Equals
        scenario = (
            Scenario("multi_step_test")
            .interact("Hello", lambda inputs: "Hi")
            .check(Equals(expected_value="Hi", key="trace.last.outputs"))
        )
        result = await scenario.run()

    For advanced usage you can instantiate with pre-filled steps or use
    ``extend()`` for a flat list of components:

        from giskard.checks import Scenario, Interact, Equals
        scenario = Scenario(
            name="multi_step_test",
            steps=[
                Step(
                    interacts=[Interact(inputs="Hello", outputs="Hi")],
                    checks=[Equals(expected_value="Hi", key="trace.last.outputs")],
                ),
            ],
        )
        result = await scenario.run()

    Attributes
    ----------
    name : str
        Scenario identifier.
    steps : list[Step]
        Steps to execute. Each step groups interacts and checks.
    trace_type : type[TraceType] | None
        Optional custom trace type to use. If not provided, the trace type will be
        inferred from the components. Useful when using custom trace subclasses
        with additional computed fields or methods.
    multiple_runs : int
        Default upper bound on how many times to execute the full scenario (each
        execution uses a fresh trace). Each run must pass for the next to run;
        execution stops on the first non-passing run (FAIL, ERROR, or SKIP). This
        is not a "retry until one success" mode.
    """

    name: str = Field(
        default="Unnamed Scenario",
        description="Scenario name",
    )
    steps: list[Step[InputType, OutputType, TraceType]] = Field(
        default_factory=list,
        description="Steps to execute. Each step groups interacts and checks.",
    )
    trace_type: type[TraceType] | None = Field(
        default=None,
        description="Type of trace to use for the scenario. If not provided, the trace type will be inferred from the components.",
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
    multiple_runs: int = Field(
        default=1,
        description=(
            "Default maximum number of full scenario executions (fresh trace per run). "
            "Each run must pass overall for another to run; stops on first non-passing run. "
            "Not a retry-until-success loop."
        ),
        ge=1,
        strict=True,
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

    def _append_step(self) -> Step[InputType, OutputType, TraceType]:
        """Append a new step."""
        step = Step(interacts=[], checks=[])
        self.steps.append(step)
        return step

    def _last_step(self) -> Step[InputType, OutputType, TraceType]:
        """Return the last step. Create one if none exists."""
        if not self.steps:
            return self._append_step()

        return self.steps[-1]

    def _ensure_step_for_interactions(self) -> Step[InputType, OutputType, TraceType]:
        """Return the last step for appending interactions. Create a new step if the last step has checks or if no step exists."""
        step = self._last_step()
        if step.checks:
            return self._append_step()
        return step

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
        """Add an interaction to the scenario.

        Creates an `Interact` with the provided inputs and outputs and adds
        it to the current step. Supports static values, callables, and generators
        just like `Interact`. If the current step already has checks, a new step
        is created first.

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
        return self.add_interaction(interaction)

    def check(self, check: Check[InputType, OutputType, TraceType]) -> Self:
        """Add a check to the scenario."""
        step = self._last_step()
        step.checks.append(check)
        return self

    def checks(self, *checks: Check[InputType, OutputType, TraceType]) -> Self:
        """Add multiple checks to the scenario."""
        step = self._last_step()
        step.checks.extend(checks)
        return self

    def add_interaction(
        self,
        interaction: InteractionSpec[InputType, OutputType, TraceType],
    ) -> Self:
        """Add a custom InteractionSpec to the scenario."""
        step = self._ensure_step_for_interactions()
        step.interacts.append(interaction)
        return self

    def add_interactions(
        self, *interactions: InteractionSpec[InputType, OutputType, TraceType]
    ) -> Self:
        """Add multiple InteractionSpec objects to the scenario."""
        step = self._ensure_step_for_interactions()
        step.interacts.extend(interactions)
        return self

    def append(
        self,
        component: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Append any component to the scenario."""
        if isinstance(component, Check):
            return self.check(component)
        return self.add_interaction(component)

    def extend(
        self,
        *components: (
            InteractionSpec[InputType, OutputType, TraceType]
            | Check[InputType, OutputType, TraceType]
        ),
    ) -> Self:
        """Extend the scenario with multiple components of any type."""
        for component in components:
            self.append(component)
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
        multiple_runs: int | None = None,
    ) -> ScenarioResult[TraceType]:
        """Execute the scenario via the default runner, with optional multiple runs.

        Each run executes all steps in order with a trace shared across those
        steps: interaction specs update the trace, then checks validate it and
        stop that run on failure. When more than one run is configured, the
        scenario is executed up to that many times, with a fresh trace each time.
        Each run must pass overall for the next to run. Execution stops early on
        the first non-passing run (FAIL, ERROR, or SKIP). Multi-run is not
        equivalent to retrying until a single passing outcome.

        Parameters
        ----------
        target : ProviderType | NotProvided
            Optional target override used to replace `NOT_PROVIDED` interaction outputs.
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.
        multiple_runs : int | None
            Optional cap on full scenario executions. When provided, it overrides
            the scenario-level `multiple_runs` value.

        Returns
        -------
        ScenarioResult
            Results from the last run executed, including multi-run metadata.
        """
        # Lazy import to avoid circular dependency
        from ..scenarios.runner import get_runner

        runner = get_runner()
        return await runner.run(
            self,
            target=target,
            return_exception=return_exception,
            multiple_runs=multiple_runs,
        )
