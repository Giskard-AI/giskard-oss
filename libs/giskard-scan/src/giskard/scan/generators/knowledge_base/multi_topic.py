"""Multi-topic knowledge-base quality scenario generator."""

import logging
from typing import Any, ClassVar, override

import numpy as np
from giskard.checks import Contradiction, LLMGenerator, Scenario, Trace
from pydantic import Field

from ..base import ScenarioContext, TargetMode
from .base import DEFAULT_KNOWLEDGE_BASE_MAX_TURNS, KnowledgeBaseScenarioGenerator

MULTI_TOPIC_MIN_DOCUMENTS = 2
MULTI_TOPIC_QUALITY_TAGS = [
    "quality:multi-topic-questions",
    "component:retrieval",
    "component:history",
]

logger = logging.getLogger(__name__)


class MultiTopicScenarioGenerator(KnowledgeBaseScenarioGenerator):
    """Generate multi-turn direct questions over different knowledge-base topics."""

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Multi Topic Questions"
    prompt_path: ClassVar[str] = "giskard.scan::scenarios/knowledge_base_multi_topic.j2"
    quality_tags: ClassVar[list[str]] = MULTI_TOPIC_QUALITY_TAGS
    max_turns: int = Field(default=DEFAULT_KNOWLEDGE_BASE_MAX_TURNS, ge=2)

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        if self._skip_for_singleturn(target_mode):
            return []

        params = self._prepare_generation_params(
            context,
            max_scenarios=max_scenarios,
            rng=rng,
            target_mode=target_mode,
        )
        if params is None:
            return []

        document_count = len(params.knowledge_base.documents)
        if document_count < MULTI_TOPIC_MIN_DOCUMENTS:
            logger.warning(
                "%s requires at least %d knowledge-base documents; skipping.",
                type(self).__name__,
                MULTI_TOPIC_MIN_DOCUMENTS,
            )
            return []

        turn_count = min(params.max_turns, document_count)
        scenarios = []
        for language in params.sampled_languages:
            turn_seed_indices = [
                int(index)
                for index in params.rng.choice(
                    document_count,
                    size=turn_count,
                    replace=False,
                )
            ]
            turn_contexts = [
                [
                    document.content
                    for document in await params.knowledge_base.closest_documents(
                        turn_seed_index, self.context_documents
                    )
                ]
                for turn_seed_index in turn_seed_indices
            ]
            scenarios.append(
                self._build_multi_topic_scenario(
                    description=context.description,
                    language=language,
                    turn_seed_indices=turn_seed_indices,
                    turn_contexts=turn_contexts,
                )
            )

        return scenarios

    def _build_multi_topic_scenario(
        self,
        description: str,
        language: str,
        turn_seed_indices: list[int],
        turn_contexts: list[list[str]],
    ) -> Scenario[Any, Any, Trace[Any, Any]]:
        joined_seed_indices = ", ".join(str(index) for index in turn_seed_indices)
        scenario = Scenario(
            f"{self.scenario_name_prefix} - Documents {joined_seed_indices}"
        )
        for turn_index, turn_context in enumerate(turn_contexts):
            scenario = scenario.interact(
                LLMGenerator(prompt_path=self.prompt_path, max_steps=1),
                metadata={"context": turn_context, "turn_index": turn_index},
            ).check(Contradiction(context_key="trace.last.metadata.context"))

        return scenario.with_annotations(
            {
                "description": description,
                "language": language,
                "seed_document_indices": turn_seed_indices,
                "reference_contexts": turn_contexts,
            }
        ).with_tags(list(self.quality_tags))
