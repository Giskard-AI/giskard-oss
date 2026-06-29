"""Public package exports for giskard.checks."""

import os
from pathlib import Path

from giskard.agents import add_prompts_path
from giskard.core.utils import get_lib_version

from . import builtin, judges
from .builtin import (
    AllOf,
    AnyOf,
    Equals,
    FnCheck,
    GreaterEquals,
    GreaterThan,
    JsonValid,
    LesserThan,
    LesserThanEquals,
    Not,
    NotEquals,
    Readability,
    RegexMatching,
    RegoPolicy,
    SemanticSimilarity,
    StringMatching,
    from_fn,
)
from .core import (
    Check,
    CheckResult,
    CheckStatus,
    GroupedSuiteResult,
    GroupStats,
    InputGenerationException,
    Interact,
    Interaction,
    InteractionSpec,
    Metric,
    Scenario,
    ScenarioResult,
    Step,
    SuiteResult,
    Target,
    TestCase,
    TestCaseResult,
    Trace,
    resolve,
)
from .core.mixin import WithEmbeddingMixin, WithGeneratorMixin
from .generators.base import BaseLLMGenerator, LLMGenerator
from .generators.dataset import DatasetInputGenerator
from .generators.user import UserSimulator
from .judges import (
    AnswerRelevance,
    BaseLLMCheck,
    Conformity,
    Contradiction,
    Groundedness,
    LLMCheckResult,
    LLMJudge,
    Toxicity,
)
from .scenarios.runner import ScenarioRunner
from .scenarios.suite import Suite
from .settings import get_default_generator, set_default_generator
from .testing import WithSpy
from .testing.runner import TestCaseRunner

__version__ = get_lib_version("giskard-checks")

# Install rich.pretty for better REPL output (including Pydantic models)
# Can be disabled by setting GISKARD_CHECKS_DISABLE_RICH_PRETTY=1
if os.getenv("GISKARD_CHECKS_DISABLE_RICH_PRETTY", "").lower() not in (
    "1",
    "true",
    "yes",
):
    from rich.pretty import install

    install()

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.checks")


__all__ = [
    "__version__",
    # Modules
    "builtin",
    "judges",
    # Core classes
    "Check",
    "CheckResult",
    "CheckStatus",
    "GroupedSuiteResult",
    "GroupStats",
    "Metric",
    "Scenario",
    "ScenarioResult",
    "Step",
    "SuiteResult",
    "Target",
    "TestCase",
    "TestCaseResult",
    "Trace",
    "resolve",
    "Interact",
    "Interaction",
    "InteractionSpec",
    "WithGeneratorMixin",
    "WithEmbeddingMixin",
    # Builtin and LLM-based checks
    "AnswerRelevance",
    "AllOf",
    "AnyOf",
    "Not",
    "BaseLLMCheck",
    "LLMCheckResult",
    "Conformity",
    "Contradiction",
    "Equals",
    "NotEquals",
    "LesserThan",
    "GreaterThan",
    "LesserThanEquals",
    "GreaterEquals",
    "FnCheck",
    "JsonValid",
    "RegoPolicy",
    "from_fn",
    "Groundedness",
    "LLMJudge",
    "Readability",
    "SemanticSimilarity",
    "Toxicity",
    "StringMatching",
    "RegexMatching",
    # Exceptions
    "InputGenerationException",
    # LLM-based generators
    "BaseLLMGenerator",
    "LLMGenerator",
    # Generators
    "DatasetInputGenerator",
    "UserSimulator",
    # Testing
    "WithSpy",
    "TestCaseRunner",
    # Scenarios
    "Suite",
    "ScenarioRunner",
    # Settings
    "set_default_generator",
    "get_default_generator",
]
