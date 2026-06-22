import numpy as np
import pytest
from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import Groundedness
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS,
    DEFAULT_KNOWLEDGE_BASE_MAX_TURNS,
    KNOWLEDGE_BASE_QUALITY_TAG,
    KnowledgeBaseScenarioGenerator,
)
from giskard.scan.utils.knowledge_base import Document, KnowledgeBase
from pydantic import ValidationError


def _knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(
        documents=(
            Document(content="alpha", embeddings=[1.0, 0.0]),
            Document(content="alpha nearby", embeddings=[0.9, 0.1]),
            Document(content="beta", embeddings=[0.0, 1.0]),
        )
    )


async def test_knowledge_base_generator_returns_empty_without_knowledge_base():
    generator = KnowledgeBaseScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(description="Support agent", languages=["en"])
    )

    assert scenarios == []


async def test_knowledge_base_generator_builds_groundedness_scenario():
    generator = KnowledgeBaseScenarioGenerator(context_documents=2, max_turns=4)

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

    assert scenario.name.startswith("Knowledge Base Question - Document ")
    assert scenario.tags == [KNOWLEDGE_BASE_QUALITY_TAG]
    assert scenario.annotations["description"] == "Support agent"
    assert scenario.annotations["language"] in {"en", "fr"}
    assert len(scenario.annotations["reference_context"]) == 2

    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert (
        interaction.inputs.prompt_path
        == "giskard.scan::scenarios/knowledge_base_question.j2"
    )
    assert interaction.inputs.max_steps == 4
    assert interaction.metadata["context"] == scenario.annotations["reference_context"]

    check = scenario.steps[0].checks[0]
    assert isinstance(check, Groundedness)
    assert check.context_key == "trace.last.metadata.context"


async def test_knowledge_base_generator_budget_subsamples_reproducibly():
    generator = KnowledgeBaseScenarioGenerator()

    first = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=2,
        rng=np.random.default_rng(42),
    )
    second = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=2,
        rng=np.random.default_rng(42),
    )

    assert len(first) == 2
    assert [scenario.name for scenario in first] == [
        scenario.name for scenario in second
    ]
    assert [scenario.annotations["language"] for scenario in first] == [
        scenario.annotations["language"] for scenario in second
    ]


async def test_knowledge_base_generator_samples_without_replacement_below_document_count():
    generator = KnowledgeBaseScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=2,
        rng=np.random.default_rng(42),
    )

    seed_indices = [
        scenario.annotations["seed_document_index"] for scenario in scenarios
    ]
    assert len(seed_indices) == 2
    assert len(set(seed_indices)) == 2


async def test_knowledge_base_generator_covers_all_documents_before_replacement():
    generator = KnowledgeBaseScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=5,
        rng=np.random.default_rng(42),
    )

    seed_indices = [
        scenario.annotations["seed_document_index"] for scenario in scenarios
    ]
    assert len(seed_indices) == 5
    assert set(seed_indices[:3]) == {0, 1, 2}


async def test_knowledge_base_generator_samples_language_per_scenario():
    generator = KnowledgeBaseScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(
            description="Support agent",
            languages=["en", "fr"],
            knowledge_base=_knowledge_base(),
        ),
        max_scenarios=4,
        rng=np.random.default_rng(42),
    )

    scenario_languages = [scenario.annotations["language"] for scenario in scenarios]
    assert len(scenario_languages) == 4
    assert set(scenario_languages) <= {"en", "fr"}


def test_knowledge_base_generator_uses_default_context_documents():
    assert (
        KnowledgeBaseScenarioGenerator().context_documents
        == DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS
    )


def test_knowledge_base_generator_uses_default_max_turns():
    assert (
        KnowledgeBaseScenarioGenerator().max_turns == DEFAULT_KNOWLEDGE_BASE_MAX_TURNS
    )


def test_knowledge_base_generator_rejects_non_positive_context_documents():
    with pytest.raises(ValidationError):
        _ = KnowledgeBaseScenarioGenerator(context_documents=0)


def test_knowledge_base_generator_rejects_non_positive_max_turns():
    with pytest.raises(ValidationError):
        _ = KnowledgeBaseScenarioGenerator(max_turns=0)
