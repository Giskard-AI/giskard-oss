from collections.abc import AsyncGenerator

import pytest
from giskard.checks import Equals, Interaction, Scenario, Trace
from giskard.checks.scenarios.runner import ScenarioRunner


def test_trace_steps_returns_empty_list_for_empty_trace():
    trace = Trace[str, str]()

    assert trace.steps() == []


def test_trace_steps_groups_single_step_without_metadata():
    trace = Trace[str, str](
        interactions=[
            Interaction(inputs="a", outputs="A"),
            Interaction(inputs="b", outputs="B"),
        ]
    )

    assert trace.steps() == [trace.interactions]


def test_trace_steps_groups_multiple_steps_by_step_index():
    first = Interaction(inputs="a", outputs="A", metadata={"step_index": 0})
    second = Interaction(inputs="b", outputs="B", metadata={"step_index": 0})
    third = Interaction(inputs="c", outputs="C", metadata={"step_index": 1})
    fourth = Interaction(inputs="d", outputs="D", metadata={"step_index": 2})

    trace = Trace[str, str](interactions=[first, second, third, fourth])

    assert trace.steps() == [
        [first, second],
        [third],
        [fourth],
    ]


@pytest.mark.asyncio
async def test_scenario_runner_tags_trace_interactions_with_step_index():
    scenario = (
        Scenario("multi_step")
        .interact("hello", "HELLO")
        .check(Equals(expected_value="HELLO", key="trace.last.outputs"))
        .interact("world", "WORLD")
    )

    result = await ScenarioRunner().run(scenario)

    assert result.final_trace.steps() == [
        [result.final_trace.interactions[0]],
        [result.final_trace.interactions[1]],
    ]
    assert result.final_trace.interactions[0].metadata["step_index"] == 0
    assert result.final_trace.interactions[1].metadata["step_index"] == 1


@pytest.mark.asyncio
async def test_scenario_runner_tags_all_generated_interactions_in_same_step():
    async def input_generator(
        trace: Trace[str, str],
    ) -> AsyncGenerator[str, Trace[str, str]]:
        yield "hello"
        yield "world"

    scenario = Scenario("generated_step").interact(
        input_generator,
        lambda inputs: inputs.upper(),
    )

    result = await ScenarioRunner().run(scenario)

    assert result.final_trace.steps() == [result.final_trace.interactions]
    assert [
        interaction.metadata["step_index"]
        for interaction in result.final_trace.interactions
    ] == [
        0,
        0,
    ]
