import pytest
from giskard.checks.core.interaction.interaction import Interaction
from giskard.checks.core.interaction.trace import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios.runner import _resolve_trace_type
from giskard.checks.utils.inference import _infer_trace_type
from giskard.core.utils import NOT_PROVIDED


class MyTrace(Trace[Interaction[str, str]], frozen=True):
    pass


class OtherTrace(Trace[Interaction[str, str]], frozen=True):
    pass


def test_infer_trace_type_returns_none_for_non_callable():
    assert _infer_trace_type("not a callable") is None


def test_infer_trace_type_returns_none_for_single_param_callable():
    def target(inputs: str) -> str:
        return inputs

    assert _infer_trace_type(target) is None


def test_infer_trace_type_returns_none_when_second_param_not_trace():
    def target(inputs: str, context: dict[str, object]) -> str:
        return inputs

    assert _infer_trace_type(target) is None


def test_infer_trace_type_returns_subclass_when_second_param_is_trace_subclass():
    def target(inputs: str, trace: MyTrace) -> str:
        return inputs

    assert _infer_trace_type(target) is MyTrace


def test_infer_trace_type_returns_base_trace_when_second_param_is_base_trace():
    def target(inputs: str, trace: Trace[Interaction[str, str]]) -> str:
        return inputs

    result = _infer_trace_type(target)
    assert result is Trace[Interaction[str, str]]


def test_infer_trace_type_works_for_callable_instance():
    class MyAgent:
        def __call__(self, inputs: str, trace: MyTrace) -> str:
            return inputs

    assert _infer_trace_type(MyAgent()) is MyTrace


def test_infer_trace_type_returns_none_for_callable_instance_no_second_param():
    class MyAgent:
        def __call__(self, inputs: str) -> str:
            return inputs

    assert _infer_trace_type(MyAgent()) is None


def test_infer_trace_type_returns_none_for_callable_with_no_annotations():
    assert _infer_trace_type(lambda x, y: x) is None


# --- _resolve_trace_type unit tests ---


def test_resolve_trace_type_returns_explicit_trace_type():
    scenario = Scenario("s", trace_type=MyTrace)

    def run_target(inputs: str, trace: OtherTrace) -> str:
        return inputs

    assert _resolve_trace_type(scenario, run_target) is MyTrace


def test_resolve_trace_type_uses_run_target_when_no_explicit():
    scenario = Scenario("s")

    def run_target(inputs: str, trace: MyTrace) -> str:
        return inputs

    assert _resolve_trace_type(scenario, run_target) is MyTrace


def test_resolve_trace_type_falls_back_to_scenario_target():
    def scenario_target(inputs: str, trace: MyTrace) -> str:
        return inputs

    scenario = Scenario("s", target=scenario_target)

    assert _resolve_trace_type(scenario, NOT_PROVIDED) is MyTrace


def test_resolve_trace_type_run_target_wins_over_scenario_target():
    def scenario_target(inputs: str, trace: MyTrace) -> str:
        return inputs

    def run_target(inputs: str, trace: OtherTrace) -> str:
        return inputs

    scenario = Scenario("s", target=scenario_target)

    assert _resolve_trace_type(scenario, run_target) is OtherTrace


def test_resolve_trace_type_defaults_to_base_trace():
    def run_target(inputs: str) -> str:
        return inputs

    scenario = Scenario("s")

    assert _resolve_trace_type(scenario, run_target) is Trace


def test_resolve_trace_type_defaults_to_base_trace_when_no_target():
    scenario = Scenario("s")
    assert _resolve_trace_type(scenario, NOT_PROVIDED) is Trace


# --- Runner integration tests ---


@pytest.mark.asyncio
async def test_runner_instantiates_inferred_trace_type():
    from giskard.checks.scenarios.runner import ScenarioRunner

    def target(inputs: str, trace: MyTrace) -> str:
        return "pong"

    scenario = Scenario("s").interact("ping")
    runner = ScenarioRunner()
    result = await runner.run(scenario, target=target)

    assert isinstance(result.final_trace, MyTrace)


@pytest.mark.asyncio
async def test_runner_explicit_trace_type_beats_inferred():
    from giskard.checks.scenarios.runner import ScenarioRunner

    def target(inputs: str, trace: OtherTrace) -> str:
        return "pong"

    scenario = Scenario("s", trace_type=MyTrace).interact("ping")
    runner = ScenarioRunner()
    result = await runner.run(scenario, target=target)

    assert isinstance(result.final_trace, MyTrace)


@pytest.mark.asyncio
async def test_runner_scenario_target_inference_used_when_no_run_target():
    from giskard.checks.scenarios.runner import ScenarioRunner

    def scenario_target(inputs: str, trace: MyTrace) -> str:
        return "pong"

    scenario = Scenario("s", target=scenario_target).interact("ping")
    runner = ScenarioRunner()
    result = await runner.run(scenario)

    assert isinstance(result.final_trace, MyTrace)
