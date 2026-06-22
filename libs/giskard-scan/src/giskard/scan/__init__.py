"""Public package exports for giskard.scan."""

from pathlib import Path

from giskard.agents import add_prompts_path
from giskard.core.utils import get_lib_version

from .catalog import generate_suite
from .generators.adversarial import AdversarialScenarioGenerator
from .generators.base import DatasetScenarioGenerator, ScenarioGenerator
from .generators.crescendo import CrescendoAttackScenarioGenerator
from .generators.goat import GOATAttackScenarioGenerator
from .generators.knowledge_base import KnowledgeBaseScenarioGenerator
from .generators.prompt_injection import PromptInjectionScenarioGenerator
from .quality import quality_scan, quality_suite_generator_registry
from .registry import SuiteGeneratorRegistry
from .utils.knowledge_base import Document, KnowledgeBase
from .vulnerability import vulnerability_scan, vulnerability_suite_generator_registry

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.scan")

__version__ = get_lib_version("giskard-scan")

__all__ = [
    "generate_suite",
    "ScenarioGenerator",
    "DatasetScenarioGenerator",
    "AdversarialScenarioGenerator",
    "CrescendoAttackScenarioGenerator",
    "GOATAttackScenarioGenerator",
    "PromptInjectionScenarioGenerator",
    "KnowledgeBaseScenarioGenerator",
    "Document",
    "KnowledgeBase",
    "SuiteGeneratorRegistry",
    "quality_suite_generator_registry",
    "quality_scan",
    "vulnerability_suite_generator_registry",
    "vulnerability_scan",
]
