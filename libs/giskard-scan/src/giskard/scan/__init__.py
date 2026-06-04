"""Public package exports for giskard.scan."""

from pathlib import Path

from giskard.agents import add_prompts_path

from .catalog import generate_suite
from .generators.adversarial import AdversarialScenarioGenerator
from .generators.base import DatasetScenarioGenerator, ScenarioGenerator
from .generators.goat import GOATAttackScenarioGenerator
from .generators.prompt_injection import PromptInjectionScenarioGenerator
from .registry import SuiteGeneratorRegistry, suite_generator_registry

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.scan")

__all__ = [
    "generate_suite",
    "ScenarioGenerator",
    "DatasetScenarioGenerator",
    "AdversarialScenarioGenerator",
    "GOATAttackScenarioGenerator",
    "PromptInjectionScenarioGenerator",
    "SuiteGeneratorRegistry",
    "suite_generator_registry",
]
