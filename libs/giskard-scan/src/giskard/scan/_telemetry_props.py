"""Aggregate, non-identifying properties for PostHog (no names, messages, or content)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .generators.base import ScenarioGenerator, TargetMode


def generator_type_counts(generators: list[ScenarioGenerator]) -> dict[str, int]:
    return dict(Counter(type(generator).__name__ for generator in generators))


def suite_scan_shape_properties(
    *,
    scan_kind: str,
    language_count: int,
    target_mode: TargetMode,
    generator_count: int,
    scenario_count: int,
    generator_types: dict[str, int],
    parallel: bool,
    max_concurrency: int | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "integration": "giskard-scan",
        "scan_kind": scan_kind,
        "language_count": language_count,
        "target_mode": target_mode,
        "generator_count": generator_count,
        "scenario_count": scenario_count,
        "generator_types": generator_types,
        "parallel": parallel,
        "max_concurrency": max_concurrency,
        **extra,
    }


def third_party_scan_shape_properties(
    *,
    tool: str,
    language_count: int | None,
    target_mode: TargetMode,
    has_probe_filter: bool,
    has_tag_filter: bool,
) -> dict[str, Any]:
    return {
        "integration": "giskard-scan",
        "scan_kind": f"third_party_{tool}",
        "tool": tool,
        "language_count": language_count,
        "target_mode": target_mode,
        "has_probe_filter": has_probe_filter,
        "has_tag_filter": has_tag_filter,
    }
