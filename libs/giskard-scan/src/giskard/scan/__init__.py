"""Public package exports for giskard.scan."""

from pathlib import Path

from giskard.agents import add_prompts_path
from giskard.core.utils import get_lib_version

from .catalog import generate_suite
from .generators.adversarial import AdversarialScenarioGenerator
from .generators.base import (
    DEFAULT_TARGET_MODE,
    LocalDatasetScenarioGenerator,
    ScenarioGenerator,
    TargetMode,
)
from .generators.crescendo import CrescendoAttackScenarioGenerator
from .generators.gcg import GCGInjectionScenarioGenerator
from .generators.goat import GOATAttackScenarioGenerator
from .generators.huggingface import HuggingFaceDatasetScenarioGenerator
from .generators.knowledge_base import (
    HallucinationScenarioGenerator,
    KnowledgeBaseScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    SycophancyScenarioGenerator,
)
from .generators.prompt_injection import PromptInjectionScenarioGenerator
from .integrations import ScanTool, list_scan_items, third_party_scan
from .quality import quality_scan, quality_suite_generator_registry
from .registry import SuiteGeneratorRegistry
from .types import ScanOptions
from .utils.knowledge_base import Document, KnowledgeBase
from .vulnerability import vulnerability_scan, vulnerability_suite_generator_registry

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.scan")

__version__ = get_lib_version("giskard-scan")

__all__ = [
    "DEFAULT_TARGET_MODE",
    "TargetMode",
    "generate_suite",
    "ScanTool",
    "list_scan_items",
    "ScenarioGenerator",
    "LocalDatasetScenarioGenerator",
    "AdversarialScenarioGenerator",
    "CrescendoAttackScenarioGenerator",
    "GCGInjectionScenarioGenerator",
    "GOATAttackScenarioGenerator",
    "HuggingFaceDatasetScenarioGenerator",
    "PromptInjectionScenarioGenerator",
    "KnowledgeBaseScenarioGenerator",
    "HallucinationScenarioGenerator",
    "MultiTopicScenarioGenerator",
    "OutOfScopeScenarioGenerator",
    "SplitQuestionsScenarioGenerator",
    "SycophancyScenarioGenerator",
    "Document",
    "KnowledgeBase",
    "ScanOptions",
    "SuiteGeneratorRegistry",
    "quality_suite_generator_registry",
    "quality_scan",
    "third_party_scan",
    "vulnerability_suite_generator_registry",
    "vulnerability_scan",
]
