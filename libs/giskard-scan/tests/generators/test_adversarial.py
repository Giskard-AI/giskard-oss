from collections.abc import Sequence
from typing import Any, override
from unittest.mock import AsyncMock, patch

from giskard.agents import BaseGenerator, GenerationParams
from giskard.checks.core import Interact, LLMChatJudge
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import Conformity
from giskard.llm.types import ChatMessage, CompletionResponse
from giskard.scan.generators.adversarial import AdversarialScenarioGenerator
from giskard.scan.generators.base import ScenarioContext


class _StubGenerator(BaseGenerator):
    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        raise NotImplementedError


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
    assert raw.inputs.generator is None
    check = scenarios[0].steps[0].checks[0]
    assert isinstance(check, Conformity)
    assert check.judge is None
    assert check.generator is None


async def test_adversarial_scenarios_forward_custom_generator():
    custom_generator = _StubGenerator()
    gen = AdversarialScenarioGenerator(generator=custom_generator)
    with patch.object(gen, "_generate_rules", new=AsyncMock(return_value=["rule1"])):
        scenarios = await gen.generate_scenario(
            ScenarioContext(description="A safety chatbot", languages=["en"]),
            max_scenarios=1,
        )
    assert len(scenarios) >= 1
    raw = scenarios[0].steps[0].interacts[0]
    assert isinstance(raw, Interact)
    assert isinstance(raw.inputs, LLMGenerator)
    assert raw.inputs.generator is custom_generator
    check = scenarios[0].steps[0].checks[0]
    assert isinstance(check, Conformity)
    assert isinstance(check.judge, LLMChatJudge)
    assert check.judge.generator is custom_generator
    assert check.generator is custom_generator
