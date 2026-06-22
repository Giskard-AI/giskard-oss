"""Knowledge-base scenario generator for document-grounded quality scans."""

from typing import Any

import numpy as np
from giskard.checks import Groundedness
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.generators import LLMGenerator
from pydantic import Field

from ..utils.knowledge_base import Document
from .base import ScenarioContext, ScenarioGenerator, TargetMode

DEFAULT_KNOWLEDGE_BASE_SCENARIOS = 5
DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS = 4
DEFAULT_KNOWLEDGE_BASE_MAX_TURNS = 3
KNOWLEDGE_BASE_QUALITY_TAG = "quality:raget"


class KnowledgeBaseScenarioGenerator(ScenarioGenerator):
    """Generate document-grounded quality scenarios from a knowledge base.

    The generator samples seed documents, retrieves their nearest neighbors,
    and builds multi-turn scenarios that use an LLM-generated user simulator
    grounded in those documents. The knowledge base is supplied run-wide via
    :class:`~giskard.scan.generators.base.ScenarioContext`; when the context
    has no knowledge base this generator produces no scenarios.
    """

    context_documents: int = Field(
        default=DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS, ge=1
    )
    max_turns: int = Field(default=DEFAULT_KNOWLEDGE_BASE_MAX_TURNS, ge=1)

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
            target_mode: Desired conversation mode. ``"singleturn"`` caps each
                scenario's user simulator to a single grounded question;
                ``"multiturn"`` (default) allows follow-up turns up to
                ``max_turns``.

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
            Scenario(f"Knowledge Base Question - Document {seed_index}")
            .interact(
                LLMGenerator(
                    prompt_path="giskard.scan::scenarios/knowledge_base_question.j2",
                    max_steps=max_turns,
                ),
                metadata={"context": context},
            )
            .check(Groundedness(context_key="trace.last.metadata.context"))
            .with_annotations(
                {
                    "description": description,
                    "language": language,
                    "seed_document_index": seed_index,
                    "reference_context": context,
                }
            )
            .with_tags([KNOWLEDGE_BASE_QUALITY_TAG])
        )
