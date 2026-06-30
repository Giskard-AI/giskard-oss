from unittest.mock import AsyncMock

import numpy as np
import pytest
from giskard.checks.core import Interact
from giskard.checks.generators import LLMGenerator
from giskard.checks.judges import Conformity, Contradiction
from giskard.scan.generators.base import ScenarioContext
from giskard.scan.generators.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS,
    DEFAULT_KNOWLEDGE_BASE_MAX_TURNS,
    HallucinationScenarioGenerator,
    KnowledgeBaseScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeCandidate,
    OutOfScopeScenarioGenerator,
    OutOfScopeValidation,
    SplitQuestionsScenarioGenerator,
)
from giskard.scan.generators.knowledge_base.hallucination import (
    DIRECT_QUESTIONS_QUALITY_TAGS,
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


def _knowledge_base_with_two_documents() -> KnowledgeBase:
    return KnowledgeBase(
        documents=(
            Document(content="alpha", embeddings=[1.0, 0.0]),
            Document(content="beta", embeddings=[0.0, 1.0]),
        )
    )


def _scenario_context() -> ScenarioContext:
    return ScenarioContext(
        description="Support agent",
        languages=["en", "fr"],
        knowledge_base=_knowledge_base(),
    )


async def test_hallucination_generator_returns_empty_without_knowledge_base():
    generator = HallucinationScenarioGenerator()

    scenarios = await generator.generate_scenario(
        ScenarioContext(description="Support agent", languages=["en"])
    )

    assert scenarios == []


async def test_hallucination_generator_builds_contradiction_scenario():
    generator = HallucinationScenarioGenerator(context_documents=2, max_turns=4)

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
        f"{HallucinationScenarioGenerator.scenario_name_prefix} - Document "
    )
    assert scenario.tags == DIRECT_QUESTIONS_QUALITY_TAGS
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
    assert isinstance(check, Contradiction)
    assert check.context_key == "trace.last.metadata.context"


async def test_hallucination_generator_budget_subsamples_reproducibly():
    generator = HallucinationScenarioGenerator()

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


async def test_hallucination_generator_samples_without_replacement_below_document_count():
    generator = HallucinationScenarioGenerator()

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


async def test_hallucination_generator_covers_all_documents_before_replacement():
    generator = HallucinationScenarioGenerator()

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


async def test_hallucination_generator_samples_language_per_scenario():
    generator = HallucinationScenarioGenerator()

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


async def test_split_questions_generator_builds_exactly_two_turn_scenario():
    generator = SplitQuestionsScenarioGenerator(context_documents=2)

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )
    scenario = scenarios[0]

    assert scenario.tags == SplitQuestionsScenarioGenerator.quality_tags
    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert (
        interaction.inputs.prompt_path
        == "giskard.scan::scenarios/knowledge_base_split_question.j2"
    )
    assert interaction.inputs.max_steps == 2
    assert isinstance(scenario.steps[0].checks[0], Contradiction)


async def test_split_questions_generator_skips_singleturn_mode():
    generator = SplitQuestionsScenarioGenerator()

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
        target_mode="singleturn",
    )

    assert scenarios == []


async def test_multi_topic_generator_checks_after_each_distinct_topic_turn():
    generator = MultiTopicScenarioGenerator(context_documents=1, max_turns=3)

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )
    scenario = scenarios[0]

    assert scenario.tags == MultiTopicScenarioGenerator.quality_tags
    assert len(scenario.steps) == 3
    assert len(set(scenario.annotations["seed_document_indices"])) == 3
    assert len(scenario.annotations["reference_contexts"]) == 3
    for turn_index, step in enumerate(scenario.steps):
        interaction = step.interacts[0]
        assert isinstance(interaction, Interact)
        assert isinstance(interaction.inputs, LLMGenerator)
        assert (
            interaction.inputs.prompt_path
            == "giskard.scan::scenarios/knowledge_base_multi_topic.j2"
        )
        assert interaction.inputs.max_steps == 1
        assert interaction.metadata["turn_index"] == turn_index
        assert (
            interaction.metadata["context"]
            == scenario.annotations["reference_contexts"][turn_index]
        )
        check = step.checks[0]
        assert isinstance(check, Contradiction)
        assert check.context_key == "trace.last.metadata.context"


async def test_multi_topic_generator_requires_at_least_two_documents():
    generator = MultiTopicScenarioGenerator()
    context = ScenarioContext(
        description="Support agent",
        languages=["en"],
        knowledge_base=KnowledgeBase(
            documents=(Document(content="alpha", embeddings=[1.0, 0.0]),)
        ),
    )

    scenarios = await generator.generate_scenario(
        context,
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )

    assert scenarios == []


async def test_multi_topic_generator_caps_turns_to_available_documents():
    generator = MultiTopicScenarioGenerator(context_documents=1)
    context = ScenarioContext(
        description="Support agent",
        languages=["en"],
        knowledge_base=_knowledge_base_with_two_documents(),
    )

    scenarios = await generator.generate_scenario(
        context,
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )
    scenario = scenarios[0]

    assert len(scenario.steps) == 2
    assert len(set(scenario.annotations["seed_document_indices"])) == 2


async def test_multi_topic_generator_skips_singleturn_mode():
    generator = MultiTopicScenarioGenerator()

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
        target_mode="singleturn",
    )

    assert scenarios == []


def test_multi_topic_generator_rejects_single_turn_budget():
    with pytest.raises(ValidationError):
        _ = MultiTopicScenarioGenerator(max_turns=1)


async def test_out_of_scope_generator_builds_conformity_scenario(
    monkeypatch: pytest.MonkeyPatch,
):
    generator = OutOfScopeScenarioGenerator(context_documents=2, max_turns=2)
    candidate = OutOfScopeCandidate(
        topic="GammaCare bundle",
        reason="It resembles the alpha products but is not present.",
    )
    validation = OutOfScopeValidation(
        is_absent=True,
        reason="Nearest documents mention alpha and beta only.",
    )
    monkeypatch.setattr(
        generator,
        "_generate_absent_candidate",
        AsyncMock(return_value=candidate),
    )
    monkeypatch.setattr(
        generator,
        "_validate_absence",
        AsyncMock(return_value=validation),
    )
    monkeypatch.setattr(
        KnowledgeBase,
        "closest_documents_to_text",
        AsyncMock(return_value=list(_knowledge_base().documents[:2])),
    )

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )
    scenario = scenarios[0]

    assert scenario.tags == OutOfScopeScenarioGenerator.quality_tags
    assert scenario.annotations["absent_topic"] == "GammaCare bundle"
    interaction = scenario.steps[0].interacts[0]
    assert isinstance(interaction, Interact)
    assert isinstance(interaction.inputs, LLMGenerator)
    assert (
        interaction.inputs.prompt_path
        == "giskard.scan::scenarios/knowledge_base_out_of_scope.j2"
    )
    assert interaction.inputs.max_steps == 2
    assert interaction.metadata["absent_topic"] == "GammaCare bundle"
    check = scenario.steps[0].checks[0]
    assert isinstance(check, Conformity)
    assert "GammaCare bundle" in check.rule
    assert "Fail only if the agent directly confirms" in check.rule
    assert "Any other response should pass" in check.rule


async def test_out_of_scope_generator_rejects_direct_text_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    generator = OutOfScopeScenarioGenerator()
    monkeypatch.setattr(
        generator,
        "_generate_absent_candidate",
        AsyncMock(
            return_value=OutOfScopeCandidate(
                topic="alpha",
                reason="Looks plausible.",
            )
        ),
    )
    validate_absence = AsyncMock(
        return_value=OutOfScopeValidation(is_absent=True, reason="unused")
    )
    monkeypatch.setattr(generator, "_validate_absence", validate_absence)

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )

    assert scenarios == []
    validate_absence.assert_not_called()


async def test_out_of_scope_generator_retries_string_matches_from_new_document(
    monkeypatch: pytest.MonkeyPatch,
):
    generator = OutOfScopeScenarioGenerator(context_documents=1)
    generate_candidate = AsyncMock(
        side_effect=[
            OutOfScopeCandidate(topic="alpha", reason="Direct match."),
            OutOfScopeCandidate(topic="GammaCare bundle", reason="Absent topic."),
        ]
    )
    monkeypatch.setattr(generator, "_generate_absent_candidate", generate_candidate)
    monkeypatch.setattr(
        generator,
        "_validate_absence",
        AsyncMock(
            return_value=OutOfScopeValidation(
                is_absent=True,
                reason="Nearest documents do not mention it.",
            )
        ),
    )
    monkeypatch.setattr(
        KnowledgeBase,
        "closest_documents_to_text",
        AsyncMock(return_value=list(_knowledge_base().documents[:1])),
    )

    scenarios = await generator.generate_scenario(
        _scenario_context(),
        max_scenarios=1,
        rng=np.random.default_rng(1),
    )

    assert len(scenarios) == 1
    assert scenarios[0].annotations["absent_topic"] == "GammaCare bundle"
    assert generate_candidate.await_count == 2


def test_knowledge_base_base_uses_default_context_documents():
    assert (
        KnowledgeBaseScenarioGenerator().context_documents
        == DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS
    )


def test_knowledge_base_base_uses_default_max_turns():
    assert (
        KnowledgeBaseScenarioGenerator().max_turns == DEFAULT_KNOWLEDGE_BASE_MAX_TURNS
    )


def test_knowledge_base_base_rejects_non_positive_context_documents():
    with pytest.raises(ValidationError):
        _ = KnowledgeBaseScenarioGenerator(context_documents=0)


def test_knowledge_base_base_rejects_non_positive_max_turns():
    with pytest.raises(ValidationError):
        _ = KnowledgeBaseScenarioGenerator(max_turns=0)
