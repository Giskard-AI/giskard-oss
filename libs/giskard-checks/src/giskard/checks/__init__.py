"""Public package exports for giskard.checks."""

import os
from pathlib import Path

from giskard.agents import add_prompts_path

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
    RegexMatching,
    SemanticSimilarity,
    StringMatching,
    from_fn,
)
from .core import (
    Check,
    CheckResult,
    CheckStatus,
    InputGenerationException,
    Interact,
    Interaction,
    InteractionSpec,
    Metric,
    Scenario,
    ScenarioResult,
    Step,
    SuiteResult,
    TestCase,
    TestCaseResult,
    Trace,
    resolve,
)
from .generators.base import BaseLLMGenerator, LLMGenerator
from .generators.user import UserSimulator
from .judges import (
    AnswerRelevance,
    BaseLLMCheck,
    Conformity,
    Groundedness,
    LLMCheckResult,
    LLMJudge,
    Toxicity,
)
from .scenarios.runner import ScenarioRunner
from .scenarios.suite import Suite
from .scenarios_generator.catalog import generate_suite
from .scenarios_generator.registry import (
    SuiteGeneratorRegistry,
    suite_generator_registry,
)
from .settings import get_default_generator, set_default_generator
from .testing import WithSpy
from .testing.runner import TestCaseRunner

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
    # Modules
    "builtin",
    "judges",
    # Core classes
    "Check",
    "CheckResult",
    "CheckStatus",
    "Metric",
    "Scenario",
    "ScenarioResult",
    "Step",
    "SuiteResult",
    "TestCase",
    "TestCaseResult",
    "Trace",
    "resolve",
    "Interact",
    "Interaction",
    "InteractionSpec",
    # Builtin and LLM-based checks
    "AnswerRelevance",
    "AllOf",
    "AnyOf",
    "Not",
    "BaseLLMCheck",
    "LLMCheckResult",
    "Conformity",
    "Equals",
    "NotEquals",
    "LesserThan",
    "GreaterThan",
    "LesserThanEquals",
    "GreaterEquals",
    "FnCheck",
    "JsonValid",
    "from_fn",
    "Groundedness",
    "LLMJudge",
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
    "UserSimulator",
    # Suite generation
    "generate_suite",
    "SuiteGeneratorRegistry",
    "suite_generator_registry",
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
