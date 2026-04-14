"""Scenario runner for executing sequences of scenario components.

This module provides a runner that executes scenarios using the handle() method
pattern, where components yield Interactions or CheckResults and receive
updated Trace objects via the async generator protocol.
"""

import time
from typing import Any, cast

from giskard.core import (
    NOT_PROVIDED,
    NotProvided,
    scoped_telemetry,
    telemetry,
    telemetry_tag,
)

from .._telemetry_props import scenario_shape_properties
from ..core import Trace
from ..core.interaction import Interact
from ..core.result import CheckResult, ScenarioResult, TestCaseResult
from ..core.scenario import Scenario, Step
from ..core.testcase import TestCase
from ..core.types import ProviderType


def _build_steps[InputType, OutputType, TraceType: Trace[Any, Any]](
    scenario: Scenario[InputType, OutputType, TraceType],
    target: (
        ProviderType[[InputType], OutputType]
        | ProviderType[[InputType, TraceType], OutputType]
        | NotProvided
    ),
) -> list[Step[InputType, OutputType, TraceType]]:
    """Build steps with target bound to Interact outputs where needed.

    If no target is provided, returns the scenario's steps as-is. Otherwise,
    returns new Step objects with interacts that have NOT_PROVIDED outputs
    replaced by the given target.
    """
    target = target if not isinstance(target, NotProvided) else scenario.target

    if isinstance(target, NotProvided):
        return scenario.steps

    steps = []
    for step in scenario.steps:
        interacts = []
        for interact in step.interacts:
            if isinstance(interact, Interact) and isinstance(
                interact.outputs, NotProvided
            ):
                interact = interact.model_copy().set_outputs(target)
            interacts.append(interact)

        steps.append(step.model_copy(update={"interacts": interacts}))

    return steps


class ScenarioRunner:
    """Execute scenarios by running their steps sequentially.

    The runner processes each step: first applies interactions to the trace,
    then runs checks against the resulting trace. Execution stops on the first
    check failure or error.

    Each step is processed as follows:
    1. **Interacts** (InteractionSpec): Add interactions to the trace.
       Specs generate interactions using their `generate()` method. Each yielded
       interaction is added to the trace, and the updated trace is sent back to
       the generator via `asend()`.
    2. **Checks**: Validate the current trace state using their `run()` method.
       If a check fails or errors, execution stops immediately.

    The runner handles exceptions from checks by converting them into
    `CheckResult.error` objects and stopping execution.

    Examples
    --------
    ```python
    runner = ScenarioRunner()
    result = await runner.run(scenario)
    ```
    """

    @scoped_telemetry
    async def run[InputType, OutputType, TraceType: Trace[Any, Any]](
        self,
        scenario: Scenario[InputType, OutputType, TraceType],
        target: (
            ProviderType[[InputType], OutputType]
            | ProviderType[[InputType, TraceType], OutputType]
            | NotProvided
        ) = NOT_PROVIDED,
        return_exception: bool = False,
    ) -> ScenarioResult[TraceType]:
        """Execute a scenario's steps sequentially with shared Trace.

        Each step is executed in order:
        - Interaction specs update the shared trace
        - Checks validate the current trace and stop execution on failure

        Execution stops on the first failing check; remaining steps are not executed.

        Parameters
        ----------
        scenario : Scenario
            The scenario to execute.
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.

        Returns
        -------
        ScenarioResult
            Results from executing the scenario.
        """

        start_time = time.perf_counter()
        telemetry_tag("giskard_component", "scenario_runner")
        telemetry_tag("giskard_operation", "scenario_run")

        trace = (
            scenario.trace_type(annotations=scenario.annotations)
            if scenario.trace_type is not None
            else cast(
                TraceType,
                Trace[InputType, OutputType](annotations=scenario.annotations),
            )
        )

        steps = _build_steps(scenario, target)
        steps_results: list[TestCaseResult] = []
        has_target = target is not NOT_PROVIDED
        shape_props = scenario_shape_properties(
            scenario,
            has_target=has_target,
        )

        _ = telemetry.capture(
            "checks_scenario_run_started",
            properties=shape_props,
        )

        for step in steps:
            trace = await trace.with_interactions(*step.interacts)

            test_case = TestCase(
                trace=trace,
                checks=step.checks,
            )
            step_result = await test_case.run(return_exception)
            steps_results.append(step_result)

            # Stop on first failure
            if not step_result.passed:
                break

        if len(steps_results) < len(steps):
            for i in range(len(steps_results), len(steps)):
                step_result = TestCaseResult(
                    results=[
                        CheckResult.skip(
                            message=f"Step {i + 1} was skipped due to previous failure"
                        )
                        for _ in steps[i].checks
                    ],
                    duration_ms=0,
                )
                steps_results.append(step_result)

        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)

        result = ScenarioResult(
            scenario_name=scenario.name,
            steps=steps_results,
            duration_ms=duration_ms,
            final_trace=trace,
        )

        _ = telemetry.capture(
            "checks_scenario_run_finished",
            properties={
                **shape_props,
                "outcome_status": result.status.value,
                "duration_ms": duration_ms,
            },
        )

        return result


_default_runner = ScenarioRunner()


def get_runner() -> ScenarioRunner:
    """Return the default process-wide `ScenarioRunner` instance.

    This function provides access to a singleton runner instance that is used
    by default when executing scenarios and test cases. The same runner instance
    is reused across all executions within a process.

    Returns
    -------
    ScenarioRunner
        The default scenario runner instance.
    """
    return _default_runner
