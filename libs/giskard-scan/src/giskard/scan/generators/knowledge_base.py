"""Knowledge-base scenario generators for document-grounded quality scans."""

from typing import Any, ClassVar, Unpack, override

import numpy as np
from giskard.checks import Contradiction, LLMGenerator, Scenario, Trace
from pydantic import ConfigDict, Field

from ..utils.knowledge_base import Document
from .base import ScenarioContext, ScenarioGenerator, TargetMode

DEFAULT_KNOWLEDGE_BASE_SCENARIOS = 5
DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS = 4
DEFAULT_KNOWLEDGE_BASE_MAX_TURNS = 3

DIRECT_QUESTIONS_QUALITY_TAGS = [
    "quality:direct-questions",
    "component:llm",
    "component:retriever",
]
SYCOPHANCY_QUALITY_TAGS = ["quality:sycophancy", "component:llm"]


class KnowledgeBaseScenarioGenerator(ScenarioGenerator):
    """Base class for document-grounded quality scenarios from a knowledge base.

    Subclasses configure the prompt, scenario name, and tags while this base
    class owns seed document sampling, nearest-neighbor retrieval, language
    sampling, turn-budget handling, and the default contradiction scenario shape.

    Attributes:
        context_documents: Maximum number of nearest-neighbor documents used as
            private reference context for each scenario.
        max_turns: Maximum number of user-simulator turns per scenario.
    """

    scenario_name_prefix: ClassVar[str] = ""
    prompt_path: ClassVar[str] = ""
    quality_tags: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.scenario_name_prefix or not cls.prompt_path or not cls.quality_tags:
            raise TypeError(
                f"Subclass {cls.__name__} must define non-empty scenario_name_prefix, prompt_path, and quality_tags."
            )

    context_documents: int = Field(
        default=DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS, ge=1
    )
    max_turns: int = Field(default=DEFAULT_KNOWLEDGE_BASE_MAX_TURNS, ge=1)

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Generate scenarios from nearest-neighbor knowledge base contexts.

        Args:
            context: Run-wide context; its ``knowledge_base`` drives sampling.
                When ``None``, an empty list is returned.
            max_scenarios: Maximum number of scenarios to generate. ``None``
                uses :data:`DEFAULT_KNOWLEDGE_BASE_SCENARIOS`.
            rng: Random generator used for reproducible seed document sampling.
            target_mode: Desired conversation mode. ``"singleturn"`` caps the
                user simulator to one turn; ``"multiturn"`` (default) allows
                follow-up turns up to ``max_turns``.

        Returns:
            Generated scenarios, or an empty list when the context carries no
            knowledge base.
        """
        knowledge_base = context.knowledge_base
        if knowledge_base is None:
            return []

        scenario_count = (
            DEFAULT_KNOWLEDGE_BASE_SCENARIOS if max_scenarios is None else max_scenarios
        )
        if scenario_count <= 0:
            return []

        effective_max_turns = self._effective_max_turns(self.max_turns, target_mode)

        rng = rng or np.random.default_rng()
        seed_indices = self._sample_seed_indices(
            len(knowledge_base.documents), scenario_count, rng
        )
        sampled_languages = self._sample_languages(
            context.languages, scenario_count, rng
        )

        scenarios = []
        for seed_index, language in zip(seed_indices, sampled_languages):
            context_documents = await knowledge_base.closest_documents(
                int(seed_index), self.context_documents
            )
            scenarios.append(
                self._build_scenario(
                    description=context.description,
                    language=language,
                    seed_index=int(seed_index),
                    context_documents=context_documents,
                    max_turns=effective_max_turns,
                )
            )

        return scenarios

    @staticmethod
    def _sample_seed_indices(
        document_count: int, scenario_count: int, rng: np.random.Generator
    ) -> list[int]:
        if scenario_count <= document_count:
            return [
                int(index)
                for index in rng.choice(
                    document_count, size=scenario_count, replace=False
                )
            ]

        covered_indices = [int(index) for index in rng.permutation(document_count)]
        extra_indices = [
            int(index)
            for index in rng.choice(
                document_count, size=scenario_count - document_count, replace=True
            )
        ]
        return covered_indices + extra_indices

    @staticmethod
    def _sample_languages(
        languages: list[str], scenario_count: int, rng: np.random.Generator
    ) -> list[str]:
        if not languages:
            return ["en"] * scenario_count
        return [
            str(language)
            for language in rng.choice(languages, size=scenario_count, replace=True)
        ]

    def _build_scenario(
        self,
        description: str,
        language: str,
        seed_index: int,
        context_documents: list[Document],
        max_turns: int,
    ) -> Scenario[Any, Any, Trace[Any, Any]]:
        context = [document.content for document in context_documents]
        return (
            Scenario(f"{self.scenario_name_prefix} - Document {seed_index}")
            .interact(
                LLMGenerator(
                    prompt_path=self.prompt_path,
                    max_steps=max_turns,
                ),
                metadata={"context": context},
            )
            .check(Contradiction(context_key="trace.last.metadata.context"))
            .with_annotations(
                {
                    "description": description,
                    "language": language,
                    "seed_document_index": seed_index,
                    "reference_context": context,
                }
            )
            .with_tags(list(self.quality_tags))
        )


class HallucinationScenarioGenerator(KnowledgeBaseScenarioGenerator):
    """Generate document-grounded hallucination quality scenarios.

    The generator samples seed documents, retrieves their nearest neighbors,
    and builds multi-turn scenarios that use an LLM-generated user simulator
    grounded in those documents. The contradiction check flags target responses
    that clearly conflict with the retrieved context.
    """

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Direct Questions"
    prompt_path: ClassVar[str] = "giskard.scan::scenarios/knowledge_base_question.j2"
    quality_tags: ClassVar[list[str]] = DIRECT_QUESTIONS_QUALITY_TAGS


class SycophancyScenarioGenerator(KnowledgeBaseScenarioGenerator):
    """Generate document-grounded sycophancy quality scenarios.

    The generator samples seed documents and nearest neighbors from the run
    knowledge base, then asks an LLM user simulator to pressure the target with
    a plausible premise explicitly contradicted by those documents.
    """

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Sycophantic Questions"
    prompt_path: ClassVar[str] = "giskard.scan::scenarios/knowledge_base_sycophancy.j2"
    quality_tags: ClassVar[list[str]] = SYCOPHANCY_QUALITY_TAGS
