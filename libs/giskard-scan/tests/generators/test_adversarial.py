from unittest.mock import AsyncMock, patch

from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.scan.generators.adversarial import AdversarialScenarioGenerator
from giskard.scan.generators.base import ScenarioContext


def test_adversarial_generator_has_max_turns_field():
    gen = AdversarialScenarioGenerator()
    assert gen.max_turns == 3


async def test_adversarial_singleturn_scenarios_use_max_steps_1():
    gen = AdversarialScenarioGenerator()
    with patch.object(gen, "_generate_rules", new=AsyncMock(return_value=["rule1"])):
        scenarios = await gen.generate_scenario(
            ScenarioContext(description="A safety chatbot", languages=["en"]),
            max_scenarios=1,
            target_mode="singleturn",
        )
    assert len(scenarios) >= 1
    raw = scenarios[0].steps[0].interacts[0]
    assert isinstance(raw, Interact)
    assert isinstance(raw.inputs, LLMGenerator)
    assert raw.inputs.max_steps == 1


async def test_adversarial_multiturn_scenarios_use_configured_max_steps():
    gen = AdversarialScenarioGenerator(max_turns=5)
    with patch.object(gen, "_generate_rules", new=AsyncMock(return_value=["rule1"])):
        scenarios = await gen.generate_scenario(
            ScenarioContext(description="A safety chatbot", languages=["en"]),
            max_scenarios=1,
            target_mode="multiturn",
        )
    assert len(scenarios) >= 1
    raw = scenarios[0].steps[0].interacts[0]
    assert isinstance(raw, Interact)
    assert isinstance(raw.inputs, LLMGenerator)
    assert raw.inputs.max_steps == 5


async def test_adversarial_default_multiturn_uses_default_max_steps():
    gen = AdversarialScenarioGenerator()
    with patch.object(gen, "_generate_rules", new=AsyncMock(return_value=["rule1"])):
        scenarios = await gen.generate_scenario(
            ScenarioContext(description="A safety chatbot", languages=["en"]),
            max_scenarios=1,
        )
    assert len(scenarios) >= 1
    raw = scenarios[0].steps[0].interacts[0]
    assert isinstance(raw, Interact)
    assert isinstance(raw.inputs, LLMGenerator)
    assert raw.inputs.max_steps == 3
