"""Out-of-scope knowledge-base quality scenario generator."""

import logging
from asyncio import TaskGroup
from typing import Any, ClassVar, override

import numpy as np
from giskard.checks import Conformity, LLMGenerator, Scenario, Trace, WithGeneratorMixin
from pydantic import BaseModel, Field

from ...utils.knowledge_base import Document, KnowledgeBase
from ..base import ScenarioContext, TargetMode
from .base import (
    KnowledgeBaseScenarioGenerator,
    _normalize_for_match,
)

OUT_OF_SCOPE_MAX_STRING_MATCH_RETRIES = 2
OUT_OF_SCOPE_QUALITY_TAGS = [
    "quality:fabricated-hallucination",
    "component:llm",
    "component:retrieval",
]

logger = logging.getLogger(__name__)


class OutOfScopeCandidate(BaseModel):
    """Absent topic/object proposed from a reference context."""

    topic: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class OutOfScopeValidation(BaseModel):
    """LLM validation result for an out-of-scope candidate."""

    is_absent: bool
    reason: str = Field(..., min_length=1)


class OutOfScopeScenarioGenerator(KnowledgeBaseScenarioGenerator, WithGeneratorMixin):
    """Generate questions about precise, plausible objects absent from the KB."""

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Out Of Scope Questions"
    prompt_path: ClassVar[str] = (
        "giskard.scan::scenarios/knowledge_base_out_of_scope.j2"
    )
    quality_tags: ClassVar[list[str]] = OUT_OF_SCOPE_QUALITY_TAGS

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        params = self._prepare_generation_params(
            context,
            max_scenarios=max_scenarios,
            rng=rng,
            target_mode=target_mode,
        )
        if params is None:
            return []

        child_rngs = params.rng.spawn(params.scenario_count)

        tasks = []
        async with TaskGroup() as task_group:
            for language, child_rng in zip(params.sampled_languages, child_rngs):
                tasks.append(
                    task_group.create_task(
                        self._generate_validated_out_of_scope_scenario(
                            knowledge_base=params.knowledge_base,
                            description=context.description,
                            language=language,
                            rng=child_rng,
                            max_turns=params.max_turns,
                        )
                    )
                )

        return [scenario for task in tasks if (scenario := task.result()) is not None]

    async def _generate_validated_out_of_scope_scenario(
        self,
        knowledge_base: KnowledgeBase,
        description: str,
        language: str,
        rng: np.random.Generator,
        max_turns: int,
    ) -> Scenario[Any, Any, Trace[Any, Any]] | None:
        knowledge_base_documents = self._document_contents(knowledge_base.documents)
        used_seed_indices: set[int] = set()
        for _ in range(OUT_OF_SCOPE_MAX_STRING_MATCH_RETRIES + 1):
            seed_index = self._sample_unused_seed_index(
                len(knowledge_base.documents), used_seed_indices, rng
            )
            used_seed_indices.add(seed_index)
            context_documents = await knowledge_base.closest_documents(
                seed_index, self.context_documents
            )
            candidate = await self._generate_absent_candidate(
                description,
                language,
                self._document_contents(context_documents),
            )
            if self._has_direct_text_match(candidate.topic, knowledge_base_documents):
                continue

            validation_documents = await knowledge_base.closest_documents_to_text(
                candidate.topic, self.context_documents
            )
            validation = await self._validate_absence(
                candidate.topic,
                candidate.reason,
                self._document_contents(validation_documents),
            )
            if not validation.is_absent:
                logger.warning(
                    "No validated out-of-scope candidate found: candidate %r "
                    + "was present in the knowledge base according to validation.",
                    candidate.topic,
                )
                continue

            return self._build_out_of_scope_scenario(
                description=description,
                language=language,
                seed_index=seed_index,
                context_documents=context_documents,
                validation_documents=validation_documents,
                candidate=candidate,
                validation=validation,
                max_turns=max_turns,
            )

        logger.warning(
            "No validated out-of-scope candidate found after %d attempts.",
            OUT_OF_SCOPE_MAX_STRING_MATCH_RETRIES + 1,
        )
        return None

    @staticmethod
    def _sample_unused_seed_index(
        document_count: int, used_seed_indices: set[int], rng: np.random.Generator
    ) -> int:
        available_indices = [
            index for index in range(document_count) if index not in used_seed_indices
        ]
        if available_indices:
            return int(rng.choice(available_indices))
        return int(rng.integers(document_count))

    async def _generate_absent_candidate(
        self, description: str, language: str, reference_context: list[str]
    ) -> OutOfScopeCandidate:
        result = (
            await self._generator.template(
                "giskard.scan::generate_suite/out_of_scope_candidate.j2"
            )
            .with_output(OutOfScopeCandidate)
            .with_inputs(
                description=description,
                language=language,
                reference_context=reference_context,
            )
            .run()
        )
        return result.output

    async def _validate_absence(
        self, topic: str, candidate_reason: str, validation_context: list[str]
    ) -> OutOfScopeValidation:
        result = (
            await self._generator.template(
                "giskard.scan::generate_suite/out_of_scope_validation.j2"
            )
            .with_output(OutOfScopeValidation)
            .with_inputs(
                topic=topic,
                candidate_reason=candidate_reason,
                validation_context=validation_context,
            )
            .run()
        )
        return result.output

    def _build_out_of_scope_scenario(
        self,
        description: str,
        language: str,
        seed_index: int,
        context_documents: list[Document],
        validation_documents: list[Document],
        candidate: OutOfScopeCandidate,
        validation: OutOfScopeValidation,
        max_turns: int,
    ) -> Scenario[Any, Any, Trace[Any, Any]]:
        reference_context = self._document_contents(context_documents)
        validation_context = self._document_contents(validation_documents)
        return (
            Scenario(f"{self.scenario_name_prefix} - Document {seed_index}")
            .interact(
                LLMGenerator(prompt_path=self.prompt_path, max_steps=max_turns),
                metadata={
                    "context": validation_context,
                    "absent_topic": candidate.topic,
                },
            )
            .check(
                Conformity(
                    rule=" ".join(
                        (
                            "Fail only if the agent directly confirms the absent",
                            f"topic/object '{candidate.topic}' exists or provides a factual",
                            "answer about it as if it exists. Any other response should pass,",
                            "including refusals, uncertainty, clarification questions,",
                            "redirects, or imperfectly phrased non-answers.",
                        )
                    )
                )
            )
            .with_annotations(
                {
                    "description": description,
                    "language": language,
                    "seed_document_index": seed_index,
                    "reference_context": reference_context,
                    "validation_context": validation_context,
                    "absent_topic": candidate.topic,
                }
            )
            .with_tags(list(self.quality_tags))
        )

    @staticmethod
    def _has_direct_text_match(topic: str, documents: list[str]) -> bool:
        normalized_topic = _normalize_for_match(topic)
        return any(
            normalized_topic and normalized_topic in _normalize_for_match(document)
            for document in documents
        )
