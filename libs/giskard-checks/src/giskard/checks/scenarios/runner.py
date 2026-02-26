"""Scenario runner for executing sequences of scenario components.

This module provides a runner that executes scenarios using the handle() method
pattern, where components yield Interactions or CheckResults and receive
updated Trace objects via the async generator protocol.
"""

from __future__ import annotations

import time
from typing import Any, cast

from ..core.check import Check
from ..core.protocols import InteractionGenerator
from ..core.result import CheckResult, ScenarioResult, TestCaseResult
from ..core.scenario import Scenario
from ..core.testcase import TestCase
from ..core.trace import Interaction, Trace


class _ScenarioStep[InputType, OutputType, TraceType: Trace]:  # pyright: ignore[reportMissingTypeArgument]
    interactions: list[
        Interaction[InputType, OutputType]
        | InteractionGenerator[Interaction[InputType, OutputType], TraceType]
    ]
    checks: list[Check[InputType, OutputType, TraceType]]

    def __init__(self):
        self.interactions = []
        self.checks = []


class _ScenarioStepsBuilder[InputType, OutputType, TraceType: Trace]:  # pyright: ignore[reportMissingTypeArgument]
    steps: list[_ScenarioStep[InputType, OutputType, TraceType]]

    def __init__(
        self,
        *sequence: Interaction[InputType, OutputType]
        | InteractionGenerator[Interaction[InputType, OutputType], TraceType]
        | Check[InputType, OutputType, TraceType],
    ):
        self.steps = []
        for component in sequence:
            if isinstance(component, Check):
                self.add_check(component)
            else:
                self.add_interaction(component)

    def add_step(self):
        self.steps.append(_ScenarioStep[InputType, OutputType, TraceType]())

    @property
    def current_step(self) -> _ScenarioStep[InputType, OutputType, TraceType]:
        if not self.steps:
            self.add_step()

        return self.steps[-1]

    def add_interaction(
        self,
        interaction: (
            Interaction[InputType, OutputType]
            | InteractionGenerator[Interaction[InputType, OutputType], TraceType]
        ),
    ):
        if len(self.current_step.checks) > 0:
            self.add_step()

        self.current_step.interactions.append(interaction)

    def add_check(self, check: Check[InputType, OutputType, TraceType]):
        self.current_step.checks.append(check)

    def build(self) -> list[_ScenarioStep[InputType, OutputType, TraceType]]:
        return self.steps


class ScenarioRunner:
    """Execute scenarios by running sequences of scenario components.

    The runner processes components sequentially, maintaining a shared Trace
    that accumulates interactions. Execution stops on the first check failure
    or error.

    Components are processed in order:
    1. **Interaction / InteractionSpec components**: Add interactions to the trace.
       Specs generate interactions using their `generate()` method. Each yielded
       interaction is added to the trace, and the updated trace is sent back to
       the generator via `asend()`.
    2. **Check components**: Validate the current trace state using their `run()`
       method. If a check fails or errors, execution stops immediately.

    The runner handles exceptions from checks by converting them into
    `CheckResult.error` objects and stopping execution.

    Examples
    --------
    ```python
    runner = ScenarioRunner()
    result = await runner.run_scenario(scenario)
    ```
    """

    async def run[InputType, OutputType, TraceType: Trace[Any, Any]](
        self,
        scenario: Scenario[InputType, OutputType, TraceType],
        return_exception: bool = False,
    ) -> ScenarioResult[InputType, OutputType]:
        """Execute a sequential scenario with shared Trace.

        Components are executed in order:
        - Interaction / InteractionSpec components update the shared trace
        - Check components validate the current trace and stop execution on failure

        Execution stops on the first failing check; remaining components are not executed.

        Parameters
        ----------
        scenario : Scenario
            The scenario to execute.
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.

        Returns
        -------
        ScenarioResult
            Results from executing the scenario components.
        """

        start_time = time.perf_counter()
        trace = (
            scenario.trace_type(annotations=scenario.annotations)
            if scenario.trace_type is not None
            else cast(
                TraceType,
                Trace[InputType, OutputType](annotations=scenario.annotations),
            )
        )
        steps = _ScenarioStepsBuilder(*scenario.sequence).build()
        steps_results: list[TestCaseResult] = []

        for step in steps:
            trace = await trace.with_interactions(*step.interactions)

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
        return ScenarioResult(
            scenario_name=scenario.name,
            steps=steps_results,
            duration_ms=int((end_time - start_time) * 1000),
            final_trace=trace,
        )


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
