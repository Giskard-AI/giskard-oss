"""Unit tests for Scenario class.

Tests cover normal cases, edge cases, and error handling for scenario execution.
"""

from collections.abc import AsyncGenerator
from typing import override

import pytest
from giskard.checks import (
    BaseInteractionSpec,
    Check,
    CheckResult,
    Interaction,
    Trace,
    scenario,
)

# Mock Components for Testing


@Check.register("mock_check")
class MockCheck(Check[str, str, Trace[str, str]]):
    """Mock check component for testing scenarios."""

    name: str | None = None
    result: CheckResult
    trace_received: Trace[str, str] | None = None

    @override
    async def run(self, trace: Trace[str, str]) -> CheckResult:
        """Execute the check."""
        self.trace_received = trace
        return self.result


@BaseInteractionSpec.register("mock_interaction_spec")
class MockInteractionSpec(BaseInteractionSpec[str, str, Trace[str, str]]):
    """Mock interaction spec component for testing scenarios."""

    interactions: list[Interaction[str, str]]
    trace_received: Trace[str, str] | None = None

    @override
    async def generate(
        self, trace: Trace[str, str]
    ) -> AsyncGenerator[Interaction[str, str], Trace[str, str]]:
        """Generate interactions."""
        self.trace_received = trace
        for interaction in self.interactions:
            trace = yield interaction


@Check.register("failing_component")
class FailingComponent(Check[str, str, Trace[str, str]]):
    """Component that raises an exception during run()."""

    error_message: str = "Test error"

    @override
    async def run(self, trace: Trace[str, str]) -> CheckResult:
        """Raise an exception."""
        raise ValueError(self.error_message)


@Check.register("named_failing_component")
class NamedFailingComponent(Check[str, str, Trace[str, str]]):
    """Failing component with a configurable name for error messages."""

    name: str | None = "my_component"
    error_message: str = "Test error"

    @override
    async def run(self, trace: Trace[str, str]) -> CheckResult:
        """Raise an exception."""
        if trace.interactions:
            return CheckResult.success()
        raise ValueError(self.error_message)


@BaseInteractionSpec.register("generator_error_component")
class GeneratorErrorComponent(BaseInteractionSpec[str, str, Trace[str, str]]):
    """Component whose generator raises an error after first yield."""

    @override
    async def generate(
        self, trace: Trace[str, str]
    ) -> AsyncGenerator[Interaction[str, str], Trace[str, str]]:
        """Yield an interaction then raise an error on next iteration."""
        yield Interaction(inputs="test", outputs="result", metadata={})
        raise RuntimeError("Generator error")


# Test Classes


class TestScenarioNormalCases:
    """Test normal execution paths for scenarios."""

    async def test_scenario_with_single_passing_check(self):
        """Test scenario with a single check that passes."""
        check = MockCheck(result=CheckResult.success(message="Check passed"))

        result = await scenario("single_check").check(check).run()

        assert result.scenario_name == "single_check"
        assert len(result.steps) == 1
        assert result.steps[0].passed
        assert result.steps[0].results[0].message == "Check passed"
        assert result.passed
        assert not result.failed
        assert not result.errored
        assert len(result.final_trace.interactions) == 0
        assert result.duration_ms >= 0
        assert check.trace_received == Trace(interactions=[])

    async def test_scenario_with_multiple_passing_checks(self):
        """Test scenario with multiple checks that all pass."""
        checks = [
            MockCheck(
                result=CheckResult.success(message=f"Check {i} passed"),
                name=f"check_{i}",
            )
            for i in range(3)
        ]
        result = await scenario("multiple_checks").checks(*checks).run()

        assert result.scenario_name == "multiple_checks"
        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        test_case_result = result.steps[0]
        assert len(test_case_result.results) == 3
        assert all(r.passed for r in test_case_result.results)
        assert result.passed
        assert not result.failed
        assert not result.errored
        assert result.duration_ms >= 0

    async def test_scenario_with_interaction_spec_only(self):
        """Test scenario with only interaction specs (no checks)."""
        interactions = [
            Interaction(inputs="input1", outputs="output1", metadata={"step": 1}),
            Interaction(inputs="input2", outputs="output2", metadata={"step": 2}),
        ]
        interaction_spec = MockInteractionSpec(interactions=interactions)
        result = await (
            scenario("interactions_only").add_interaction_spec(interaction_spec).run()
        )

        assert result.scenario_name == "interactions_only"
        assert len(result.steps) == 1
        test_case_result = result.steps[0]
        assert len(test_case_result.results) == 0
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions == interactions
        assert result.passed  # No checks means passed
        assert not result.failed
        assert not result.errored

    async def test_scenario_with_direct_interactions_only(self):
        """Test scenario with direct Interaction objects (no InteractionSpec, no checks)."""
        interaction1 = Interaction(
            inputs="input1", outputs="output1", metadata={"step": 1}
        )
        interaction2 = Interaction(
            inputs="input2", outputs="output2", metadata={"step": 2}
        )
        result = await (
            scenario("direct_interactions_only")
            .add_interaction(interaction1)
            .add_interaction(interaction2)
            .run()
        )

        assert result.scenario_name == "direct_interactions_only"
        assert len(result.steps) == 1
        test_case_result = result.steps[0]
        assert len(test_case_result.results) == 0
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.passed  # No checks means passed
        assert not result.failed
        assert not result.errored

    async def test_scenario_with_interactions_and_checks(self):
        """Test scenario with interaction specs and checks mixed."""
        interaction1 = Interaction(inputs="input1", outputs="output1")
        interaction2 = Interaction(inputs="input2", outputs="output2")
        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])

        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await (
            scenario("mixed_scenario")
            .add_interaction_spec(interaction_spec1)
            .check(check1)
            .add_interaction_spec(interaction_spec2)
            .check(check2)
            .run()
        )

        assert result.scenario_name == "mixed_scenario"
        assert len(result.steps) == 2
        assert all(r.passed for r in result.steps)
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.passed

        # Verify check2 received trace with both interactions
        assert check2.trace_received is not None
        assert len(check2.trace_received.interactions) == 2

    async def test_scenario_with_direct_interactions_and_checks(self):
        """Test scenario with direct Interaction objects and checks mixed."""
        interaction1 = Interaction(inputs="input1", outputs="output1")
        interaction2 = Interaction(inputs="input2", outputs="output2")

        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await (
            scenario("direct_interactions_and_checks")
            .add_interaction(interaction1)
            .check(check1)
            .add_interaction(interaction2)
            .check(check2)
            .run()
        )

        assert result.scenario_name == "direct_interactions_and_checks"
        assert len(result.steps) == 2
        assert all(r.passed for r in result.steps)
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.passed

        # Verify check2 received trace with both interactions
        assert check2.trace_received is not None
        assert len(check2.trace_received.interactions) == 2
        assert check2.trace_received.interactions[0] == interaction1
        assert check2.trace_received.interactions[1] == interaction2

    async def test_scenario_with_mixed_interaction_types(self):
        """Test scenario with both direct Interaction objects and InteractionSpec objects."""
        direct_interaction = Interaction(inputs="direct", outputs="direct_output")
        interaction_spec_interaction = Interaction(inputs="spec", outputs="spec_output")
        interaction_spec = MockInteractionSpec(
            interactions=[interaction_spec_interaction]
        )

        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await (
            scenario("mixed_interaction_types")
            .add_interaction(direct_interaction)
            .check(check1)
            .add_interaction_spec(interaction_spec)
            .check(check2)
            .run()
        )

        assert result.scenario_name == "mixed_interaction_types"
        assert len(result.steps) == 2
        assert all(r.passed for r in result.steps)
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == direct_interaction
        assert result.final_trace.interactions[1] == interaction_spec_interaction
        assert result.passed

        # Verify check1 received trace with first interaction
        assert check1.trace_received is not None
        assert len(check1.trace_received.interactions) == 1
        assert check1.trace_received.interactions[0] == direct_interaction

        # Verify check2 received trace with both interactions
        assert check2.trace_received is not None
        assert len(check2.trace_received.interactions) == 2
        assert check2.trace_received.interactions[0] == direct_interaction
        assert check2.trace_received.interactions[1] == interaction_spec_interaction

    async def test_scenario_all_checks_run_in_step_despite_failure(self):
        """Test that all checks in a step run even if one fails."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1 passed"))
        check2 = MockCheck(result=CheckResult.failure(message="Check 2 failed"))
        check3 = MockCheck(result=CheckResult.success(message="Check 3 passed"))

        result = await (
            scenario("all_checks_run").check(check1).check(check2).check(check3).run()
        )

        assert result.scenario_name == "all_checks_run"
        assert len(result.steps) == 1  # All checks in same step
        assert len(result.steps[0].results) == 3  # All checks run even if one fails
        assert result.steps[0].results[0].passed
        assert result.steps[0].results[1].failed
        assert result.steps[0].results[2].passed
        assert result.steps[0].failed  # Step failed because check2 failed
        assert result.failed
        assert not result.passed
        assert not result.errored

    async def test_scenario_all_checks_run_in_step_despite_error(self):
        """Test that all checks in a step run even if one errors."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1 passed"))
        check2 = MockCheck(result=CheckResult.error(message="Check 2 errored"))
        check3 = MockCheck(result=CheckResult.success(message="Check 3 passed"))

        result = await (
            scenario("all_checks_run_despite_error")
            .check(check1)
            .check(check2)
            .check(check3)
            .run()
        )

        assert result.scenario_name == "all_checks_run_despite_error"
        assert len(result.steps) == 1  # All checks in same step
        assert len(result.steps[0].results) == 3  # All checks run even if one errors
        assert result.steps[0].results[0].passed
        assert result.steps[0].results[1].errored
        assert result.steps[0].results[2].passed
        assert result.steps[0].errored  # Step errored because check2 errored
        assert result.errored
        assert not result.passed
        assert not result.failed

    async def test_scenario_skips_subsequent_steps_on_failure(self):
        """Test that scenario skips subsequent steps when a step fails."""
        interaction1 = Interaction(inputs="input1", outputs="output1")
        interaction2 = Interaction(inputs="input2", outputs="output2")
        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])

        check1 = MockCheck(result=CheckResult.failure(message="Check 1 failed"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2 passed"))

        result = await (
            scenario("skips_steps_on_failure")
            .add_interaction_spec(interaction_spec1)
            .check(check1)
            .add_interaction_spec(interaction_spec2)
            .check(check2)
            .run()
        )

        assert result.scenario_name == "skips_steps_on_failure"
        assert len(result.steps) == 2
        # First step failed
        assert result.steps[0].failed
        assert len(result.steps[0].results) == 1
        assert result.steps[0].results[0].failed
        # Second step was skipped
        assert result.steps[1].skipped
        assert len(result.steps[1].results) == 1
        assert result.steps[1].results[0].skipped
        assert result.steps[1].results[0].message is not None
        assert "skipped due to previous failure" in result.steps[1].results[0].message
        # Only first interaction was added to trace
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0] == interaction1
        assert result.failed

    async def test_scenario_skips_subsequent_steps_on_error(self):
        """Test that scenario skips subsequent steps when a step errors."""
        interaction1 = Interaction(inputs="input1", outputs="output1")
        interaction2 = Interaction(inputs="input2", outputs="output2")
        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])

        check1 = MockCheck(result=CheckResult.error(message="Check 1 errored"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2 passed"))

        result = await (
            scenario("skips_steps_on_error")
            .add_interaction_spec(interaction_spec1)
            .check(check1)
            .add_interaction_spec(interaction_spec2)
            .check(check2)
            .run()
        )

        assert result.scenario_name == "skips_steps_on_error"
        assert len(result.steps) == 2
        # First step errored
        assert result.steps[0].errored
        assert len(result.steps[0].results) == 1
        assert result.steps[0].results[0].errored
        # Second step was skipped
        assert result.steps[1].skipped
        assert len(result.steps[1].results) == 1
        assert result.steps[1].results[0].skipped
        assert result.steps[1].results[0].message is not None
        assert "skipped due to previous failure" in result.steps[1].results[0].message
        # Only first interaction was added to trace
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0] == interaction1
        assert result.errored

    async def test_trace_accumulation_across_components(self):
        """Test that trace accumulates interactions across components."""
        interaction1 = Interaction(inputs="1", outputs="2")
        interaction2 = Interaction(inputs="3", outputs="4")
        interaction3 = Interaction(inputs="5", outputs="6")

        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        check1 = MockCheck(result=CheckResult.success())
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])
        check2 = MockCheck(result=CheckResult.success())
        interaction_spec3 = MockInteractionSpec(interactions=[interaction3])

        result = await (
            scenario("trace_accumulation")
            .add_interaction_spec(interaction_spec1)
            .check(check1)
            .add_interaction_spec(interaction_spec2)
            .check(check2)
            .add_interaction_spec(interaction_spec3)
            .run()
        )

        assert len(result.final_trace.interactions) == 3
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.final_trace.interactions[2] == interaction3

        # Verify check2 received trace with first two interactions
        assert check2.trace_received is not None
        assert len(check2.trace_received.interactions) == 2
        assert check2.trace_received.interactions[0] == interaction1
        assert check2.trace_received.interactions[1] == interaction2

    async def test_check_receives_updated_trace(self):
        """Test that checks receive the trace with all previous interactions."""
        interaction1 = Interaction(inputs="a", outputs="b")
        interaction2 = Interaction(inputs="c", outputs="d")

        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("check_receives_trace")
            .add_interaction_spec(interaction_spec1)
            .add_interaction_spec(interaction_spec2)
            .check(check)
            .run()
        )

        assert result.passed
        # Verify check received trace with both interactions
        assert check.trace_received is not None
        assert len(check.trace_received.interactions) == 2
        assert check.trace_received.interactions[0] == interaction1
        assert check.trace_received.interactions[1] == interaction2

    async def test_scenario_with_skipped_checks(self):
        """Test scenario with skipped checks (should continue execution)."""
        check1 = MockCheck(result=CheckResult.skip(message="Check 1 skipped"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2 passed"))
        check3 = MockCheck(result=CheckResult.skip(message="Check 3 skipped"))

        result = await (
            scenario("skipped_checks").check(check1).check(check2).check(check3).run()
        )

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 3
        assert result.steps[0].results[0].skipped
        assert result.steps[0].results[1].passed
        assert result.steps[0].results[2].skipped
        # Scenario should be marked as passed when at least one check passed and no checks errored or failed
        assert result.passed
        assert not result.failed
        assert not result.errored
        assert not result.skipped  # Not all skipped

    async def test_scenario_all_checks_skipped(self):
        """Test scenario where all checks are skipped."""
        checks = [
            MockCheck(result=CheckResult.skip(message=f"Check {i} skipped"))
            for i in range(3)
        ]
        result = await scenario("all_skipped").checks(*checks).run()

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 3
        assert all(r.skipped for r in result.steps[0].results)
        assert result.steps[0].skipped
        assert result.skipped
        assert not result.passed


class TestScenarioEdgeCases:
    """Test edge cases for scenarios."""

    async def test_scenario_with_empty_sequence(self):
        """Test scenario with empty sequence."""
        result = await scenario("empty_sequence").run()

        assert result.scenario_name == "empty_sequence"
        assert len(result.steps) == 0
        assert len(result.final_trace.interactions) == 0
        # Empty sequence with no steps: all([]) is True, but we want False
        # The property should handle empty steps list
        assert result.passed  # Empty steps list means passed (all([]) is True)
        assert not result.failed
        assert not result.errored
        assert not result.skipped

    async def test_scenario_with_single_component(self):
        """Test scenario with single component."""
        check = MockCheck(result=CheckResult.success(message="Single check"))
        result = await scenario("single_component").check(check).run()

        assert len(result.steps) == 1
        assert result.steps[0].passed
        assert result.passed

    async def test_scenario_with_only_interactions(self):
        """Test scenario with only interactions, no checks."""
        interactions = [
            Interaction(inputs=str(i), outputs=str(i * 2), metadata={"index": i})
            for i in range(5)
        ]
        interaction_spec = MockInteractionSpec(interactions=interactions)
        result = await (
            scenario("only_interactions").add_interaction_spec(interaction_spec).run()
        )

        assert (
            len(result.steps) == 1
        )  # Step created for interactions even without checks
        assert len(result.steps[0].results) == 0  # No check results
        assert len(result.final_trace.interactions) == 5
        assert result.passed  # No checks means passed

    async def test_scenario_with_only_direct_interactions(self):
        """Test scenario with only direct Interaction objects, no InteractionSpec, no checks."""
        interactions = [
            Interaction(inputs=str(i), outputs=str(i * 2), metadata={"index": i})
            for i in range(5)
        ]
        result = await (
            scenario("only_direct_interactions").add_interactions(*interactions).run()
        )

        assert (
            len(result.steps) == 1
        )  # Step created for interactions even without checks
        assert len(result.steps[0].results) == 0  # No check results
        assert len(result.final_trace.interactions) == 5
        assert result.final_trace.interactions == interactions
        assert result.passed  # No checks means passed

    async def test_scenario_with_only_checks(self):
        """Test scenario with only checks, no interactions."""
        checks = [
            MockCheck(result=CheckResult.success(message=f"Check {i}"))
            for i in range(5)
        ]
        result = await scenario("only_checks").checks(*checks).run()

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 5
        assert len(result.final_trace.interactions) == 0
        assert result.passed

    async def test_scenario_with_multiple_consecutive_interactions(self):
        """Test scenario with multiple consecutive interaction specs."""
        interaction_specs: list[BaseInteractionSpec[str, str, Trace[str, str]]] = [
            MockInteractionSpec(
                interactions=[Interaction(inputs=str(i), outputs=str(i * 2))]
            )
            for i in range(3)
        ]
        result = await (
            scenario("consecutive_interactions")
            .add_interaction_specs(*interaction_specs)
            .run()
        )

        assert (
            len(result.steps) == 1
        )  # All consecutive interactions grouped into one step
        assert len(result.final_trace.interactions) == 3
        interaction_inputs = [
            interaction.inputs for interaction in result.final_trace.interactions
        ]
        assert interaction_inputs == ["0", "1", "2"]

    async def test_scenario_with_multiple_consecutive_direct_interactions(self):
        """Test scenario with multiple consecutive direct Interaction objects."""
        interactions = [
            Interaction(inputs=str(i), outputs=str(i * 2)) for i in range(3)
        ]
        result = await (
            scenario("consecutive_direct_interactions")
            .add_interactions(*interactions)
            .run()
        )

        assert (
            len(result.steps) == 1
        )  # All consecutive interactions grouped into one step
        assert len(result.final_trace.interactions) == 3
        assert result.final_trace.interactions == interactions
        interaction_inputs = [
            interaction.inputs for interaction in result.final_trace.interactions
        ]
        assert interaction_inputs == ["0", "1", "2"]

    async def test_scenario_with_multiple_consecutive_checks(self):
        """Test scenario with multiple consecutive checks."""
        checks = [
            MockCheck(result=CheckResult.success(message=f"Check {i}"))
            for i in range(3)
        ]
        result = await scenario("consecutive_checks").checks(*checks).run()

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 3
        assert all(r.passed for r in result.steps[0].results)
        assert result.steps[0].passed

    async def test_scenario_with_large_sequence(self):
        """Test scenario with a larger sequence of components."""
        builder = scenario("large_sequence")
        for i in range(10):
            if i % 2 == 0:
                builder.add_interaction_spec(
                    MockInteractionSpec(
                        interactions=[Interaction(inputs=str(i), outputs=str(i * 2))]
                    )
                )
            else:
                builder.check(
                    MockCheck(result=CheckResult.success(message=f"Check {i}"))
                )

        result = await builder.run()

        # Pattern: interaction, check, interaction, check, ...
        # This creates separate steps when interactions come before checks
        # Step 1: interaction0, check1
        # Step 2: interaction2, check3
        # Step 3: interaction4, check5
        # Step 4: interaction6, check7
        # Step 5: interaction8, check9
        assert len(result.steps) == 5  # 5 steps (each with interaction + check)
        assert len(result.final_trace.interactions) == 5  # 5 interactions
        assert result.passed


class TestScenarioErrorHandling:
    """Test error handling in scenarios."""

    async def test_component_raises_exception(self):
        """Test that component exceptions are converted to CheckResult.ERROR and all checks run."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1 passed"))
        failing_component = FailingComponent(error_message="Component failed")
        check2 = MockCheck(result=CheckResult.success(message="Check 2 passed"))

        builder = (
            scenario("component_exception")
            .check(check1)
            .check(failing_component)
            .check(check2)
        )

        with pytest.raises(ValueError, match="Component failed"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 3  # All checks run even if one errors
        assert result.steps[0].results[0].passed
        assert result.steps[0].results[1].errored
        assert result.steps[0].results[2].passed
        error_message = result.steps[0].results[1].message
        assert error_message is not None and "Component failed" in error_message
        assert result.steps[0].results[1].details["check_kind"] == "failing_component"
        assert result.steps[0].errored
        assert result.errored
        assert not result.passed

    async def test_component_with_name_raises_exception(self):
        """Test exception handling when component has a name."""
        failing_component = NamedFailingComponent(
            name="custom_component", error_message="Named component failed"
        )

        builder = scenario("named_component_exception").check(failing_component)

        with pytest.raises(ValueError, match="Named component failed"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        assert len(result.steps) == 1
        assert result.steps[0].errored
        named_error_message = result.steps[0].results[0].message
        assert (
            named_error_message is not None
            and "custom_component" in named_error_message
        )
        # Error message format: "Check 'custom_component' failed with error: Named component failed"
        assert "custom_component" in named_error_message
        assert "Named component failed" in named_error_message
        assert result.steps[0].results[0].errored

    async def test_all_checks_run_in_step_despite_exception(self):
        """Test that all checks in a step run even if one raises an exception."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        failing_component = FailingComponent(error_message="Error occurred")
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))
        check3 = MockCheck(result=CheckResult.success(message="Check 3"))

        builder = (
            scenario("all_checks_run_despite_exception")
            .check(check1)
            .check(failing_component)
            .check(check2)
            .check(check3)
        )

        with pytest.raises(ValueError, match="Error occurred"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        assert len(result.steps) == 1  # All consecutive checks grouped into one step
        assert len(result.steps[0].results) == 4  # All checks run even if one errors
        assert result.steps[0].results[0].passed
        assert result.steps[0].results[1].errored
        assert result.steps[0].results[2].passed
        assert result.steps[0].results[3].passed
        assert result.steps[0].errored
        assert result.errored

    async def test_generator_raises_exception_after_yield(self):
        """Test that generator exceptions from InteractionSpec propagate."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        generator_error_component = GeneratorErrorComponent()
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        # InteractionSpec generator errors currently propagate and stop execution
        # The first interaction should be added before the error is raised
        with pytest.raises(RuntimeError, match="Generator error"):
            _ = await (
                scenario("generator_exception")
                .check(check1)
                .add_interaction_spec(generator_error_component)
                .check(check2)
                .run()
            )

    async def test_all_executed_results_collected(self):
        """Test that all checks in a step are executed and results collected."""
        checks = [
            MockCheck(result=CheckResult.success(message=f"Check {i}"))
            for i in range(3)
        ]
        failing_component = FailingComponent(error_message="Error")
        check4 = MockCheck(result=CheckResult.success(message="Check 4"))

        builder = (
            scenario("collects_all_results")
            .check(checks[0])
            .check(checks[1])
            .check(checks[2])
            .check(failing_component)
            .check(check4)
        )

        with pytest.raises(ValueError, match="Error"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        # All consecutive checks grouped into one step
        assert len(result.steps) == 1
        assert len(result.steps[0].results) == 5  # All checks run even if one errors
        assert all(r.passed for r in result.steps[0].results[:3])
        assert result.steps[0].results[3].errored
        assert result.steps[0].results[4].passed
        assert result.steps[0].errored

    async def test_error_result_contains_traceback(self):
        """Test that error results contain traceback information."""
        failing_component = FailingComponent(error_message="Test error")

        builder = scenario("error_traceback").check(failing_component)

        with pytest.raises(ValueError, match="Test error"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        assert len(result.steps) == 1
        assert result.steps[0].results[0].errored
        assert "traceback" in result.steps[0].results[0].details
        assert "Test error" in result.steps[0].results[0].details["traceback"]

    async def test_error_with_interactions_preserves_trace(self):
        """Test that errors preserve trace state up to that point."""
        interaction1 = Interaction(inputs="a", outputs="b")
        interaction2 = Interaction(inputs="c", outputs="d")

        interaction_spec1 = MockInteractionSpec(interactions=[interaction1])
        interaction_spec2 = MockInteractionSpec(interactions=[interaction2])
        failing_component = FailingComponent(error_message="Error after interactions")

        builder = (
            scenario("error_preserves_trace")
            .add_interaction_spec(interaction_spec1)
            .add_interaction_spec(interaction_spec2)
            .check(failing_component)
        )

        with pytest.raises(ValueError, match="Error after interactions"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        # Trace should contain both interactions even though error occurred
        # All components (interactions + check) are grouped into one step
        assert len(result.steps) == 1
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.errored

    async def test_error_with_direct_interactions_preserves_trace(self):
        """Test that errors preserve trace state when using direct Interaction objects."""
        interaction1 = Interaction(inputs="a", outputs="b")
        interaction2 = Interaction(inputs="c", outputs="d")
        failing_component = FailingComponent(
            error_message="Error after direct interactions"
        )

        builder = (
            scenario("error_preserves_trace_direct")
            .add_interaction(interaction1)
            .add_interaction(interaction2)
            .check(failing_component)
        )

        with pytest.raises(ValueError, match="Error after direct interactions"):
            _ = await builder.run()

        # Test with return_exception=True to return the exception
        result = await builder.run(return_exception=True)

        # Trace should contain both interactions even though error occurred
        # All components (interactions + check) are grouped into one step
        assert len(result.steps) == 1
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert result.errored

    async def test_duration_calculated_correctly(self):
        """Test that scenario duration is calculated and non-negative."""
        check = MockCheck(result=CheckResult.success(message="Test check"))
        result = await scenario("duration_test").check(check).run()

        # Duration should be non-negative
        assert result.duration_ms >= 0
        # Duration should be reasonable (less than 1 second for a simple check)
        assert result.duration_ms < 1000

    async def test_append_with_interaction(self):
        """Test that append() method works with Interaction objects."""
        interaction = Interaction(inputs="Hello", outputs="Hi", metadata={"test": True})
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("add_interaction_test").append(interaction).append(check).run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0] == interaction
        assert result.final_trace.interactions[0].metadata == {"test": True}

    async def test_append_with_interaction_spec(self):
        """Test that append() method works with BaseInteractionSpec objects."""
        interaction = Interaction(inputs="Hello", outputs="Hi")
        interaction_spec = MockInteractionSpec(interactions=[interaction])
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("add_interaction_spec_test")
            .append(interaction_spec)
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0] == interaction

    async def test_append_with_check(self):
        """Test that append() method works with Check objects."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await scenario("add_check_test").append(check1).append(check2).run()

        assert result.passed
        assert len(result.steps) == 1
        assert len(result.steps[0].results) == 2

    async def test_extend_with_mixed_components(self):
        """Test that extend() method works with mixed component types."""
        interaction1 = Interaction(inputs="Hello", outputs="Hi")
        interaction2 = Interaction(inputs="How are you?", outputs="Good")
        interaction_spec = MockInteractionSpec(interactions=[interaction2])
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await (
            scenario("adds_mixed_test")
            .extend(interaction1, interaction_spec, check1, check2)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 2
        assert result.final_trace.interactions[0] == interaction1
        assert result.final_trace.interactions[1] == interaction2
        assert (
            len(result.steps) == 1
        )  # All consecutive interactions grouped, then checks
        assert len(result.steps[0].results) == 2  # Both checks run

    async def test_extend_with_multiple_checks(self):
        """Test that extend() method works with multiple checks."""
        checks = [
            MockCheck(result=CheckResult.success(message=f"Check {i}"))
            for i in range(3)
        ]

        result = await scenario("adds_multiple_checks_test").extend(*checks).run()

        assert result.passed
        assert len(result.steps) == 1
        assert len(result.steps[0].results) == 3

    async def test_extend_with_multiple_interactions(self):
        """Test that extend() method works with multiple Interaction objects."""
        interactions = [
            Interaction(inputs=str(i), outputs=str(i * 2)) for i in range(3)
        ]

        result = await (
            scenario("adds_multiple_interactions_test").extend(*interactions).run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 3
        assert result.final_trace.interactions == interactions

    async def test_extend_with_empty_args(self):
        """Test that extend() method works with no arguments."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("adds_empty_test").extend().append(check).run()  # No arguments
        )

        assert result.passed
        assert len(result.steps) == 1
        assert len(result.steps[0].results) == 1

    async def test_append_and_extend_chaining(self):
        """Test that append() and extend() can be chained together."""
        interaction1 = Interaction(inputs="Hello", outputs="Hi")
        interaction2 = Interaction(inputs="How are you?", outputs="Good")
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))
        check3 = MockCheck(result=CheckResult.success(message="Check 3"))

        result = await (
            scenario("add_adds_chain_test")
            .append(interaction1)
            .extend(check1, check2)
            .append(interaction2)
            .extend(check3)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 2
        assert len(result.steps) == 2  # interaction1+checks, then interaction2+check
        assert len(result.steps[0].results) == 2
        assert len(result.steps[1].results) == 1

    async def test_interact_with_static_values(self):
        """Test that interact() method works with static input/output values."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_static_test").interact("Hello", "Hi").append(check).run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0].inputs == "Hello"
        assert result.final_trace.interactions[0].outputs == "Hi"

    async def test_interact_with_metadata(self):
        """Test that interact() method accepts optional metadata."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_metadata_test")
            .interact("Hello", "Hi", metadata={"source": "test", "index": 1})
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0].inputs == "Hello"
        assert result.final_trace.interactions[0].outputs == "Hi"
        assert result.final_trace.interactions[0].metadata == {
            "source": "test",
            "index": 1,
        }

    async def test_interact_multiple_times(self):
        """Test that interact() can be called multiple times."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_multiple_test")
            .interact("Hello", "Hi")
            .interact("How are you?", "Good")
            .interact("What's up?", "Nothing much")
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 3
        assert result.final_trace.interactions[0].inputs == "Hello"
        assert result.final_trace.interactions[0].outputs == "Hi"
        assert result.final_trace.interactions[1].inputs == "How are you?"
        assert result.final_trace.interactions[1].outputs == "Good"
        assert result.final_trace.interactions[2].inputs == "What's up?"
        assert result.final_trace.interactions[2].outputs == "Nothing much"

    async def test_interact_without_metadata(self):
        """Test that interact() works without metadata (defaults to empty dict)."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_no_metadata_test")
            .interact("Hello", "Hi")
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0].metadata == {}

    async def test_interact_chained_with_checks(self):
        """Test that interact() can be chained with checks."""
        check1 = MockCheck(result=CheckResult.success(message="Check 1"))
        check2 = MockCheck(result=CheckResult.success(message="Check 2"))

        result = await (
            scenario("interact_checks_chain_test")
            .interact("Hello", "Hi")
            .append(check1)
            .interact("How are you?", "Good")
            .append(check2)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 2
        assert len(result.steps) == 2  # interact+check, then interact+check
        assert len(result.steps[0].results) == 1
        assert len(result.steps[1].results) == 1

    async def test_interact_with_callable_outputs(self):
        """Test that interact() works with callable outputs."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_callable_outputs_test")
            .interact("Hello", lambda inputs: f"Echo: {inputs.upper()}")
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0].inputs == "Hello"
        assert result.final_trace.interactions[0].outputs == "Echo: HELLO"

    async def test_interact_with_trace_dependent_inputs(self):
        """Test that interact() works with trace-dependent inputs."""
        check = MockCheck(result=CheckResult.success())

        result = await (
            scenario("interact_trace_dependent_test")
            .interact(
                lambda trace: f"Message #{len(trace.interactions) + 1}",
                lambda inputs, trace: f"Received: {inputs}",
            )
            .append(check)
            .run()
        )

        assert result.passed
        assert len(result.final_trace.interactions) == 1
        assert result.final_trace.interactions[0].inputs == "Message #1"
        assert result.final_trace.interactions[0].outputs == "Received: Message #1"
