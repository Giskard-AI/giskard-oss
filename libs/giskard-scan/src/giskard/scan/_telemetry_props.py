"""Shape properties for PostHog scan telemetry (no user content or secrets)."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from giskard.checks import SuiteResult
from giskard.core import telemetry_capture, telemetry_run_context, telemetry_tag

from .generators.base import ScenarioGenerator, TargetMode


def generator_type_counts(generators: list[ScenarioGenerator]) -> dict[str, int]:
    return dict(Counter(type(generator).__name__ for generator in generators))


def scan_shape_properties(
    *,
    scan_type: str,
    languages: list[str],
    target_mode: TargetMode,
    max_scenarios: int | None = None,
    seed: int | None = None,
    group_by: str | None = None,
    parallel: bool | None = None,
    max_concurrency: int | None = None,
    generator_count: int | None = None,
    generator_types: dict[str, int] | None = None,
    scenario_count: int | None = None,
    knowledge_base_document_count: int | None = None,
    commercial_use: bool | None = None,
    third_party_probes: list[str] | None = None,
    third_party_tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "integration": "giskard-scan",
        "scan_type": scan_type,
        "languages": languages,
        "target_mode": target_mode,
        "max_scenarios": max_scenarios,
        "seed": seed,
        "group_by": group_by,
        "parallel": parallel,
        "max_concurrency": max_concurrency,
        "generator_count": generator_count,
        "generator_types": generator_types,
        "scenario_count": scenario_count,
        "knowledge_base_document_count": knowledge_base_document_count,
        "commercial_use": commercial_use,
        "third_party_probes": third_party_probes,
        "third_party_tags": third_party_tags,
    }


def suite_result_properties(result: SuiteResult) -> dict[str, Any]:
    return {
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "errored_count": result.errored_count,
        "skipped_count": result.skipped_count,
        "scenario_count": len(result.results),
    }


@contextmanager
def scan_telemetry_scope(
    scan_type: str, shape_props: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Open a telemetry scope for a scan run and capture started/finished events."""
    with telemetry_run_context():
        telemetry_tag("giskard_component", "scan")
        telemetry_tag("giskard_operation", "scan_run")
        telemetry_tag("scan_type", scan_type)
        telemetry_capture("scan_run_started", properties=shape_props)
        start_time = time.perf_counter()
        finished_props: dict[str, Any] = {}
        try:
            yield finished_props
        finally:
            finished_props.setdefault(
                "duration_ms", int((time.perf_counter() - start_time) * 1000)
            )
            telemetry_capture(
                "scan_run_finished",
                properties={**shape_props, **finished_props},
            )
