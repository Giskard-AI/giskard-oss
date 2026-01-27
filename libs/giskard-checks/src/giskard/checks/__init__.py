"""Public package exports for giskard.checks."""

from pathlib import Path

from giskard.agents import add_prompts_path

from . import builtin
from .builtin import (
    BaseLLMCheck,
    Conformity,
    Equals,
    FnCheck,
    GreaterEquals,
    GreaterThan,
    Groundedness,
    LesserThan,
    LesserThanEquals,
    LLMCheckResult,
    LLMJudge,
    NotEquals,
    StringMatching,
    from_fn,
)
from .core import (
    Check,
    CheckResult,
    CheckStatus,
    Interaction,
    Metric,
    Scenario,
    ScenarioResult,
    TestCase,
    TestCaseResult,
    Trace,
)
from .core.interaction import BaseInteractionSpec
from .generators.user import UserSimulator
from .interaction import InteractionSpec
from .scenarios.builder import ScenarioBuilder, scenario
from .scenarios.runner import ScenarioRunner
from .settings import get_default_generator, set_default_generator
from .testing import WithSpy
from .testing.runner import TestCaseRunner

add_prompts_path(str(Path(__file__).parent / "prompts"), "giskard.checks")


__all__ = [
    # Modules
    "builtin",
    # Core classes
    "Check",
    "CheckResult",
    "CheckStatus",
    "Metric",
    "Scenario",
    "ScenarioResult",
    "TestCase",
    "TestCaseResult",
    "Trace",
    "Interaction",
    "BaseInteractionSpec",
    # Builtin checks
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
    "from_fn",
    "Groundedness",
    "LLMJudge",
    "StringMatching",
    # Interaction
    "InteractionSpec",
    # Generators
    "UserSimulator",
    # Testing
    "WithSpy",
    "TestCaseRunner",
    # Scenarios
    "ScenarioBuilder",
    "scenario",
    "ScenarioRunner",
    # Settings
    "set_default_generator",
    "get_default_generator",
]
