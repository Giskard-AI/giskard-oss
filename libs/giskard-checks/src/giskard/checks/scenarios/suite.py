import time
from typing import Any, Generic, Self, TypeVar

from giskard.core import telemetry_capture, telemetry_run_context, telemetry_tag
from giskard.core.utils import NOT_PROVIDED, NotProvided
from pydantic import BaseModel, Field

from .._telemetry_props import suite_shape_properties
from ..core.interaction import Trace
from ..core.result import ScenarioResult, SuiteResult
from ..core.scenario import Scenario
from ..core.types import ProviderType

InputType = TypeVar("InputType", infer_variance=True)
OutputType = TypeVar("OutputType", infer_variance=True)


class Suite(BaseModel, Generic[InputType, OutputType]):
    """A suite of scenarios that can be run together with a shared target.

    A suite holds multiple scenarios and can run them serially, optionally
    binding a single target SUT to all scenarios at once. If a target is
    provided at the suite level or during the run call, it overrides any
    scenario-level target.

    Attributes
    ----------
    name : str
        Suite identifier.
    scenarios : list[Scenario]
        List of scenarios to execute.
    target : Any | NotProvided
        Optional suite-level target SUT.

    Examples
    --------
    ```python
    from giskard.checks import Suite, Scenario

    scenario1 = Scenario("scenario_1").interact("hello")
    scenario2 = Scenario("scenario_2").interact("hi")

    suite = Suite(name="my_suite", target=my_sut)
    suite.append(scenario1).append(scenario2)

    result = await suite.run()
    print(result.pass_rate)
    ```
    """

    name: str = Field(..., description="Suite name")
    scenarios: list[Scenario[InputType, OutputType, Trace[Any, Any]]] = Field(
        default_factory=list, description="Scenarios in the suite"
    )
    target: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, Trace[Any, Any]], OutputType]
        | NotProvided
    ) = Field(
        default=NOT_PROVIDED,
        description="Suite-level target SUT that will override any scenario-level target.",
    )

    def append(
        self,
        scenario: Scenario[InputType, OutputType, Trace[Any, Any]],
    ) -> Self:
        """Add a scenario to the suite.

        Parameters
        ----------
        scenario : Scenario
            The scenario to add to the suite.

        Returns
        -------
        Suite
            The suite itself, allowing fluent chaining.
        """
        self.scenarios.append(scenario)
        return self

    async def run(
        self,
        target: (
            ProviderType[[InputType], OutputType]
            | ProviderType[
                [InputType, Trace[Any, Any]], OutputType
            ]  # Trace[Any, Any] because scenarios in suite have different TraceType
            | NotProvided
        ) = NOT_PROVIDED,
        return_exception: bool = False,
    ) -> SuiteResult:
        """Run all scenarios in the suite serially.

        Parameters
        ----------
        target : Any | NotProvided
            Override target for all scenarios in the suite. If provided, this
            overrides both the suite-level target and any scenario-level targets.
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.

        Returns
        -------
        SuiteResult
            Aggregated results from all scenarios.

        Examples
        --------
        ```python
        from giskard.checks import Suite

        suite = Suite(name="my_suite", target=my_sut_v1)
        suite.append(scenario_1).append(scenario_2)
        result_v1 = await suite.run()
        result_v2 = await suite.run(target=my_sut_v2)
        ```
        """
        results: list[ScenarioResult[Trace[Any, Any]]] = []

        target = target if not isinstance(target, NotProvided) else self.target
        has_target = not isinstance(target, NotProvided)

        with telemetry_run_context():
            telemetry_tag("giskard_component", "suite")
            telemetry_tag("giskard_operation", "suite_run")

            shape_props = suite_shape_properties(
                scenario_count=len(self.scenarios),
                has_target=has_target,
            )
            telemetry_capture(
                "checks_suite_run_started",
                properties=shape_props,
            )

            start_time = time.perf_counter()
            for scenario in self.scenarios:
                result = await scenario.run(
                    target=target, return_exception=return_exception
                )
                results.append(result)
            end_time = time.perf_counter()

            suite_result = SuiteResult(
                results=results,
                duration_ms=int((end_time - start_time) * 1000),
            )

            telemetry_capture(
                "checks_suite_run_finished",
                properties={
                    **shape_props,
                    "duration_ms": suite_result.duration_ms,
                    "passed_count": suite_result.passed_count,
                    "failed_count": suite_result.failed_count,
                    "errored_count": suite_result.errored_count,
                    "skipped_count": suite_result.skipped_count,
                },
            )

        return suite_result
