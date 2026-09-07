import numpy as np
from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import Contradiction
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.knowledge_base import SycophancyScenarioGenerator
from giskard.scan.generators.knowledge_base.sycophancy import SYCOPHANCY_QUALITY_TAGS
from giskard.scan.utils.knowledge_base import Document, KnowledgeBase


def _knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(
        documents=(
            Document(content="alpha", embeddings=[1.0, 0.0]),
            Document(content="alpha nearby", embeddings=[0.9, 0.1]),
            Document(content="beta", embeddings=[0.0, 1.0]),
        )
    )


async def test_sycophancy_generator_returns_empty_without_knowledge_base():
    generator = SycophancyScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(description="Support agent", languages=["en"])
    )

    assert scenarios == []


async def test_sycophancy_generator_builds_contradiction_scenario():
    generator = SycophancyScenarioGenerator(context_documents=2, max_turns=4)

    scenarios = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en", "fr"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )
    scenario = scenarios[0]

    assert scenario.name.startswith(
        f"{SycophancyScenarioGenerator.scenario_name_prefix} - Document "
    )
    assert scenario.tags == SYCOPHANCY_QUALITY_TAGS
    assert scenario.annotations["description"] == "Support agent"
    assert scenario.annotations["language"] in {"en", "fr"}
    assert len(scenario.annotations["reference_context"]) == 2

    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert (
        interaction.inputs.prompt_path
        == "giskard.scan::scenarios/knowledge_base_sycophancy.j2"
    )
    assert interaction.inputs.max_steps == 4
    assert interaction.metadata["context"] == scenario.annotations["reference_context"]

    check = scenario.steps[0].checks[0]
    assert isinstance(check, Contradiction)
    assert check.context_key == "trace.last.metadata.context"


async def test_sycophancy_generator_respects_singleturn_mode():
    generator = SycophancyScenarioGenerator(max_turns=4)

    scenarios = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=1,
        rng=np.random.default_rng(1),
        target_mode="singleturn",
    )

    interaction = scenarios[0].steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert interaction.inputs.max_steps == 1
