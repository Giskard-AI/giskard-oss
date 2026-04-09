"""Aggregate, non-identifying properties for PostHog (no names, messages, or content)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .core.check import Check
from .core.scenario import Scenario, Step

_DEFAULT_SCENARIO_NAME = "Unnamed Scenario"


def check_kind_counts_from_steps(steps: list[Step[Any, Any, Any]]) -> dict[str, int]:
    kinds = [getattr(c, "kind", "unknown") for step in steps for c in step.checks]
    return dict(Counter(kinds))


def check_kind_counts_from_sequence(
    checks: list[Check[Any, Any, Any]],
) -> dict[str, int]:
    kinds = [getattr(c, "kind", "unknown") for c in checks]
    return dict(Counter(kinds))


def scenario_shape_properties(
    scenario: Scenario[Any, Any, Any],
    *,
    has_target: bool,
) -> dict[str, Any]:
    total_interacts = sum(len(s.interacts) for s in scenario.steps)
    total_checks = sum(len(s.checks) for s in scenario.steps)
    return {
        "integration": "giskard-checks",
        "step_count": len(scenario.steps),
        "total_interaction_specs": total_interacts,
        "total_checks": total_checks,
        "has_target": has_target,
        "uses_custom_trace_type": scenario.trace_type is not None,
        "has_scenario_annotations": bool(scenario.annotations),
        "uses_custom_scenario_name": scenario.name != _DEFAULT_SCENARIO_NAME,
        "check_kinds": check_kind_counts_from_steps(scenario.steps),
    }


def test_case_shape_properties(
    *,
    check_count: int,
    trace_interaction_count: int,
    has_trace_annotations: bool,
    has_test_case_name: bool,
    check_kinds: dict[str, int],
) -> dict[str, Any]:
    return {
        "integration": "giskard-checks",
        "check_count": check_count,
        "trace_interaction_count": trace_interaction_count,
        "has_trace_annotations": has_trace_annotations,
        "has_test_case_name": has_test_case_name,
        "check_kinds": check_kinds,
    }


def suite_shape_properties(
    *,
    scenario_count: int,
    has_target: bool,
) -> dict[str, Any]:
    return {
        "integration": "giskard-checks",
        "scenario_count": scenario_count,
        "has_target": has_target,
    }
