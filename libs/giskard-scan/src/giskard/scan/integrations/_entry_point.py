"""Top-level entry point for third-party scanner integrations."""

import time
from typing import Any, Literal

from giskard.checks import SuiteResult, Target, Trace
from giskard.core import telemetry_capture, telemetry_run_context, telemetry_tag

from .._telemetry_props import third_party_scan_shape_properties
from ..generators.base import TargetMode


async def third_party_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    tool: Literal["garak", "lidar"],
    *,
    description: str,
    languages: list[str] | None = None,
    **kwargs: Any,
) -> SuiteResult:
    """Run an external security scanner against a Giskard target.

    Args:
        target: Agent or provider target to evaluate.
        tool: Scanner to use. ``"garak"`` or ``"lidar"``.
        description: Natural-language description of the agent under test.
            For lidar this becomes the ``TargetInfo.agent_description`` that
            probes use to build their attacks. Garak has no target-profile
            concept and ignores it.
        languages: BCP-47 language codes the agent handles. For lidar this
            becomes ``TargetInfo.languages``. Garak ignores it.
        **kwargs: Tool-specific options. For garak:
            ``probes: list[str] | None`` restricts which probes run; omitted
            means all active loadable probes, while an empty list runs none.
            ``target_mode: str`` (default ``"multiturn"``) skips garak
            iterative probes when set to ``"singleturn"``.
          For lidar:
            ``probes: list[str] | None`` restricts which probes run by id
            (e.g. ``"deepset-injection:1.0"``); ``None`` runs all.
            ``tags: list[str] | None`` restricts probes by tag; ``None`` runs
            all. ``target_mode: str`` (default ``"multiturn"``) skips lidar's
            multi-turn probes (crescendo, goat, ...) when set to
            ``"singleturn"``.

    Returns:
        The completed suite result.
    """
    target_mode: TargetMode = kwargs.get("target_mode", "multiturn")
    shape_props = third_party_scan_shape_properties(
        tool=tool,
        language_count=len(languages) if languages is not None else None,
        target_mode=target_mode,
        has_probe_filter=kwargs.get("probes") is not None,
        has_tag_filter=kwargs.get("tags") is not None,
    )

    with telemetry_run_context():
        telemetry_tag("giskard_component", "scan_third_party")
        telemetry_tag("giskard_operation", "third_party_scan")
        telemetry_capture("scan_third_party_run_started", properties=shape_props)

        start_time = time.perf_counter()
        if tool == "garak":
            from .garak import GarakScanAdapter

            # Garak has no TargetInfo concept: drop the context args so its
            # run() signature is unaffected.
            result = await GarakScanAdapter().run(target, **kwargs)
        elif tool == "lidar":
            from .lidar import LidarScanAdapter

            result = await LidarScanAdapter().run(
                target, description=description, languages=languages, **kwargs
            )
        else:
            raise ValueError(f"Unknown tool {tool!r}. Available: ['garak', 'lidar']")
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        telemetry_capture(
            "scan_third_party_run_finished",
            properties={
                **shape_props,
                "duration_ms": duration_ms,
                "scenario_count": len(result.results),
                "passed_count": result.passed_count,
                "failed_count": result.failed_count,
                "errored_count": result.errored_count,
                "skipped_count": result.skipped_count,
            },
        )

    return result
