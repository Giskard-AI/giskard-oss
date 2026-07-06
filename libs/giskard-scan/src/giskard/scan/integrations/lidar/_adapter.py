"""Adapter that runs the private lidar scanner through Giskard scan."""

import logging
import time
from importlib.util import find_spec
from typing import Any

from giskard.checks import (
    CheckResult,
    Interaction,
    Metric,
    ScenarioResult,
    SuiteResult,
    Target,
    TestCaseResult,
    Trace,
    get_default_generator,
)
from giskard.llm.types import ChatMessage

from .._shared import reject_unexpected_kwargs

logger = logging.getLogger(__name__)


def lidar_available() -> bool:
    """Return True if the private lidar dependency is importable."""
    return find_spec("lidar") is not None


def _require_lidar() -> None:
    if not lidar_available():
        raise ImportError(
            "lidar is not installed. It is a private Giskard package and is not "
            "available on PyPI. Install it from git: "
            "pip install git+https://github.com/Giskard-AI/lidar.git@v0.2.7"
        )


async def _trace_from_messages(messages: list[ChatMessage]) -> Trace:  # pyright: ignore[reportMissingTypeArgument]
    """Rebuild a scan Trace from a lidar attempt's flat message list.

    Lidar owns its own executor and hands back finished conversations, so we
    reconstruct a display trace by pairing each user turn with the assistant
    reply that follows it. System/tool messages carry no input/output pair and
    are skipped. A trailing user message with no reply yields outputs=None.
    """
    interactions: list[Interaction] = []  # pyright: ignore[reportMissingTypeArgument]
    pending_input: str | None = None
    for message in messages:
        if message.role == "user":
            if pending_input is not None:
                interactions.append(Interaction(inputs=pending_input, outputs=None))
            pending_input = message.content  # pyright: ignore[reportAssignmentType]
        elif message.role == "assistant":
            if pending_input is not None:
                interactions.append(
                    Interaction(inputs=pending_input, outputs=message.content)
                )
                pending_input = None
        # system / tool messages: no input/output pairing, skip
    if pending_input is not None:
        interactions.append(Interaction(inputs=pending_input, outputs=None))
    return await Trace.from_interactions(*interactions)


_SEVERITY_SCORE = {
    "safe": 0.0,
    "minor": 0.33,
    "major": 0.66,
    "critical": 1.0,
}


def _severity_label(severity: object) -> "str | None":
    """Normalize a lidar Severity enum (or None) to its string label."""
    if severity is None:
        return None
    return getattr(severity, "value", str(severity))


class LidarScanAdapter:
    """Build and run a Giskard suite from a lidar scan."""

    async def run[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        target: Target[InputType, OutputType, TraceType],
        *,
        description: str,
        languages: list[str] | None = None,
        **kwargs: Any,
    ) -> SuiteResult:
        """Run a lidar scan against ``target`` and return a scan SuiteResult.

        ``description`` and ``languages`` build the ``TargetInfo`` that lidar
        probes read to construct their attacks. ``discover_target_info`` stays
        False: the target profile now comes from the caller, so lidar never
        needs to run its own discovery pass. ``agent_name`` / ``company_name`` /
        ``competitors`` are left at their None/[] defaults; probes that read them
        only use them as prompt-template values and degrade gracefully.
        """
        _require_lidar()
        from lidar import run_scan
        from lidar.resources.target_info import TargetInfo

        from ._target import ScanTargetGenerator

        probes = kwargs.pop("probes", None)
        tags = kwargs.pop("tags", None)
        if kwargs.pop("target_mode", None) is not None:
            # TODO: filter out multiturn probes if target_mode is singleturn
            logger.debug("target_mode is ignored for lidar; lidar owns turn logic.")
        reject_unexpected_kwargs("lidar", kwargs)

        target_info = TargetInfo(
            agent_description=description,
            languages=languages or [],
        )

        bridged = ScanTargetGenerator(target=target)
        start = time.perf_counter()
        # run_scan returns a ScanRun immediately (scan runs in the background); await
        # wait_for_completion() to block, then read .scan_result off it. Failures
        # propagate: an empty suite would read as "scan succeeded, no vulnerabilities".
        scan_run = await run_scan(
            target=bridged,  # pyright: ignore[reportArgumentType]
            target_info=target_info,
            probe_ids=probes,
            tags_filter=tags,
            generator=get_default_generator(),
            discover_target_info=False,
            base_seed=None,
        )
        await scan_run.wait_for_completion()
        scan_result = scan_run.scan_result

        duration_ms = int((time.perf_counter() - start) * 1000)
        if scan_result is None:  # completed but produced nothing -> empty suite
            return SuiteResult(results=[], duration_ms=duration_ms)
        return await self._to_suite_result(scan_result, duration_ms, bridged)

    async def _trace_from_target_calls(self, attempt, bridged) -> Trace:  # pyright: ignore[reportMissingTypeArgument]
        """Rebuild a scan Trace by joining lidar's per-call records to the typed
        interactions the bridge captured.

        ``attempt.target_calls`` is lidar's authoritative, ordered list of
        ``target.complete`` calls for this attempt; each carries the ``call_id``
        the bridge keyed its structured ``Interaction`` on. Joining on that id
        preserves the real typed inputs/outputs instead of re-pairing flattened
        ``attempt.messages`` strings. When the attempt made no target calls
        (e.g. the probe errored before reaching the target), fall back to the
        message reconstruction so a partial transcript still displays.
        """
        target_calls = getattr(attempt, "target_calls", None) or []
        if not target_calls:
            return await _trace_from_messages(attempt.messages)
        interactions: list[Interaction] = []  # pyright: ignore[reportMissingTypeArgument]
        for target_call in target_calls:
            interaction = bridged._by_call_id.get(target_call.call_id)
            if interaction is not None:
                interactions.append(interaction)
        return await Trace.from_interactions(*interactions)

    async def _to_suite_result(
        self, scan_result, duration_ms: int, bridged
    ) -> SuiteResult:
        scenario_results: list[ScenarioResult] = []  # pyright: ignore[reportMissingTypeArgument]
        for execution in scan_result.results:
            probe_info = execution.probe_info
            result = execution.result
            # Errored / skipped probes carry no ProbeResult (result is None):
            # surface them as a single visible error/skip scenario, don't crash.
            if result is None:
                check = self._execution_to_check(execution)
                scenario_results.append(
                    ScenarioResult(
                        scenario_name=f"Lidar {probe_info.name}",
                        steps=[TestCaseResult(results=[check], duration_ms=0)],
                        # ScenarioResult.final_trace is a required Trace field
                        # (no None allowed); an errored probe never produced any
                        # interactions, so use an empty trace as the "no trace" value.
                        final_trace=await Trace.from_interactions(),
                        tags=list(probe_info.tags),
                        duration_ms=0,
                    )
                )
                continue
            for attempt_idx, attempt in enumerate(result.attempts):
                check = self._attempt_to_check(probe_info, attempt)
                scenario_results.append(
                    ScenarioResult(
                        scenario_name=f"Lidar {probe_info.name} #{attempt_idx + 1}",
                        steps=[TestCaseResult(results=[check], duration_ms=0)],
                        final_trace=await self._trace_from_target_calls(
                            attempt, bridged
                        ),
                        tags=list(probe_info.tags),
                        duration_ms=0,
                    )
                )
        return SuiteResult(results=scenario_results, duration_ms=duration_ms)

    def _execution_to_check(self, execution) -> CheckResult:
        """Map an errored/skipped ProbeExecution (result is None) to a check."""
        probe_info = execution.probe_info
        details: dict[str, object] = {
            "check_name": probe_info.name,
            "probe_id": probe_info.id,
        }
        status = str(getattr(execution, "status", "")).lower()
        message = str(execution.error) if execution.error is not None else status
        if "skip" in status:
            return CheckResult.skip(message=message or "skipped", details=details)
        return CheckResult.error(message=message or "errored", details=details)

    def _attempt_to_check(self, probe_info, attempt) -> CheckResult:
        label = _severity_label(attempt.severity)
        details: dict[str, object] = {
            "check_name": probe_info.name,
            "probe_id": probe_info.id,
        }
        if label is not None:
            details["severity"] = label
        # Carry lidar's per-attempt evidence dict (eval_result, objective,
        # injected payloads, ...) so nothing the probe produced is dropped.
        if getattr(attempt, "metadata", None):
            details["metadata"] = dict(attempt.metadata)
        # Only emit a metric for a severity we have a score for. An unknown
        # severity label (e.g. lidar adds a new level) degrades gracefully to
        # no metric rather than crashing the whole probe's scenario.
        score = _SEVERITY_SCORE.get(label) if label is not None else None
        metrics = (
            [Metric(name=probe_info.name, value=score)] if score is not None else []
        )

        if attempt.error is not None:
            return CheckResult.error(message=attempt.reason, details=details)
        # Polarity flip: lidar "successful" means the ATTACK succeeded, which is a
        # vulnerability => scan failure.
        if attempt.successful:
            return CheckResult.failure(
                message=attempt.reason, details=details, metrics=metrics
            )
        return CheckResult.success(
            message=attempt.reason, details=details, metrics=metrics
        )
