"""Quality scenario generator implementations for giskard.scan."""

from .base import (
    DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS,
    DEFAULT_KNOWLEDGE_BASE_MAX_TURNS,
    DEFAULT_KNOWLEDGE_BASE_SCENARIOS,
    KnowledgeBaseScenarioGenerator,
)
from .hallucination import HallucinationScenarioGenerator
from .multi_topic import MultiTopicScenarioGenerator
from .out_of_scope import (
    OutOfScopeCandidate,
    OutOfScopeScenarioGenerator,
    OutOfScopeValidation,
)
from .split_questions import SplitQuestionsScenarioGenerator
from .sycophancy import SycophancyScenarioGenerator

__all__ = [
    "DEFAULT_KNOWLEDGE_BASE_CONTEXT_DOCUMENTS",
    "DEFAULT_KNOWLEDGE_BASE_MAX_TURNS",
    "DEFAULT_KNOWLEDGE_BASE_SCENARIOS",
    "HallucinationScenarioGenerator",
    "KnowledgeBaseScenarioGenerator",
    "MultiTopicScenarioGenerator",
    "OutOfScopeCandidate",
    "OutOfScopeScenarioGenerator",
    "OutOfScopeValidation",
    "SplitQuestionsScenarioGenerator",
    "SycophancyScenarioGenerator",
]
