"""Knowledge-base sycophancy quality scenario generator."""

from typing import ClassVar

from .base import KnowledgeBaseScenarioGenerator

SYCOPHANCY_QUALITY_TAGS = ["quality:sycophancy-hallucinations", "component:llm"]


class SycophancyScenarioGenerator(KnowledgeBaseScenarioGenerator):
    """Generate document-grounded sycophancy quality scenarios.

    The generator samples seed documents and nearest neighbors from the run
    knowledge base, then asks an LLM user simulator to pressure the target with
    a plausible premise explicitly contradicted by those documents.
    """

    scenario_name_prefix: ClassVar[str] = "Knowledge Base Sycophantic Questions"
    prompt_path: ClassVar[str] = "giskard.scan::scenarios/knowledge_base_sycophancy.j2"
    quality_tags: ClassVar[list[str]] = SYCOPHANCY_QUALITY_TAGS
