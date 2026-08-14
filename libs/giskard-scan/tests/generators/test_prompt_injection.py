from giskard.agents.templates import get_prompts_manager
from giskard.checks import Interaction, Trace
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.prompt_injection import (
    PromptInjectionScenarioGenerator,
)

_OUTPUT_SCHEMA_MARKER = "OUTPUT_SCHEMA_MARKER"
_PROHIBITED_OBJECTIVE_RULE = (
    "The underlying objective must request illegal or dangerous activity"
)


def _scenario_by_technique(scenarios, technique):
    return next(
        scenario
        for scenario in scenarios
        if scenario.annotations.get("technique") == technique
    )


async def test_prompt_injection_generator_loads_scenarios():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )
    assert len(scenarios) == 18


async def test_prompt_injection_generator_injects_annotations():
    gen = PromptInjectionScenarioGenerator()
    description = "A customer support chatbot"
    languages = ["en", "fr"]
    scenarios = await gen.generate_scenario(
        ScenarioContext(description=description, languages=languages)
    )
    for scenario in scenarios:
        assert scenario.annotations.get("description") == description
        assert set(scenario.annotations.get("languages", [])) == set(languages)


async def test_multilingual_injection_uses_dedicated_template():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )
    scenario = _scenario_by_technique(scenarios, "multilingual")

    interact = scenario.steps[0].interacts[0]
    assert (
        interact.inputs.prompt_path
        == "giskard.scan::scenarios/llm01_multilingual_injection.j2"
    )
    assert "target languages" not in scenario.annotations["technique_instruction"]


async def test_refusal_suppression_uses_adaptive_template():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )
    scenario = _scenario_by_technique(scenarios, "refusal_suppression")

    interact = scenario.steps[0].interacts[0]
    assert (
        interact.inputs.prompt_path
        == "giskard.scan::scenarios/llm01_refusal_suppression.j2"
    )


async def test_obfuscation_techniques_request_direct_compliance():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )

    for technique in ("leet_speak", "encoding"):
        scenario = _scenario_by_technique(scenarios, technique)
        instruction = scenario.annotations["technique_instruction"]
        rule = scenario.steps[0].checks[0].rule
        assert "Do not ask the agent to" in instruction
        assert "comply with the prohibited instruction" in rule


async def test_prompt_injection_templates_render_all_branches():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"])
    )
    prompts_manager = get_prompts_manager()

    for scenario in scenarios:
        annotations = scenario.annotations
        traces = (
            Trace[str, str](annotations=annotations),
            Trace[str, str](
                interactions=[
                    Interaction(
                        inputs="Initial prohibited request",
                        outputs="I cannot comply with that request.",
                    )
                ],
                annotations=annotations,
            ),
        )
        for interact in scenario.steps[0].interacts:
            for trace in traces:
                messages = await prompts_manager.render_template(
                    interact.inputs.prompt_path,
                    {
                        "trace": trace,
                        "_instr_output": _OUTPUT_SCHEMA_MARKER,
                    },
                )
                assert len(messages) == 1
                assert messages[0].role == "user"
                assert messages[0].content
                assert _OUTPUT_SCHEMA_MARKER in messages[0].content
                assert _PROHIBITED_OBJECTIVE_RULE in messages[0].content

    refusal = _scenario_by_technique(scenarios, "refusal_suppression")
    refusal_trace = Trace[str, str](
        interactions=[
            Interaction(
                inputs="Initial prohibited request",
                outputs="I cannot comply with that request.",
            )
        ],
        annotations=refusal.annotations,
    )
    refusal_messages = await prompts_manager.render_template(
        refusal.steps[0].interacts[0].inputs.prompt_path,
        {
            "trace": refusal_trace,
            "_instr_output": _OUTPUT_SCHEMA_MARKER,
        },
    )
    refusal_prompt = refusal_messages[0].content
    assert "keep exactly the same prohibited objective" in refusal_prompt
    assert "exact refusal wording, rationale, or safety pattern" in refusal_prompt


def _max_steps(scenario):
    return [
        interact.inputs.max_steps
        for step in scenario.steps
        for interact in step.interacts
        if hasattr(interact.inputs, "max_steps")
    ]


async def test_prompt_injection_multiturn_keeps_dataset_max_steps():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"]),
        target_mode="multiturn",
    )
    # The bundled LLM01 scenario encodes a multi-step interaction.
    assert any(steps > 1 for scenario in scenarios for steps in _max_steps(scenario))


async def test_prompt_injection_singleturn_caps_max_steps_to_1():
    gen = PromptInjectionScenarioGenerator()
    scenarios = await gen.generate_scenario(
        ScenarioContext(description="A documentation chatbot", languages=["en"]),
        target_mode="singleturn",
    )
    all_steps = [steps for scenario in scenarios for steps in _max_steps(scenario)]
    assert all_steps  # sanity: the dataset has interaction generators
    assert all(steps == 1 for steps in all_steps)
