"""Direct knowledge-base question quality scenario generator."""

from typing import ClassVar

from .base import KnowledgeBaseScenarioGenerator

DIRECT_QUESTIONS_QUALITY_TAGS = [
    "quality:direct-hallucination",
    "component:llm",
]


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
