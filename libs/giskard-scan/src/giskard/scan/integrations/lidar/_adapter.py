"""Adapter that runs the private lidar scanner through Giskard scan."""

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

from giskard.checks import (
    CheckResult,
    Interaction,
    Metric,
    ScenarioResult,
    SuiteResult,
    TestCaseResult,
    Trace,
)

if TYPE_CHECKING:
    # Type-only: lidar's compat Message union. Never imported at runtime — the
    # function only reads .role/.content, so no lidar dependency is introduced.
    from lidar.giskard_compat import Message

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


async def _trace_from_messages(messages: "list[Message]") -> Trace:
    """Rebuild a scan Trace from a lidar attempt's flat message list.

    Lidar owns its own executor and hands back finished conversations, so we
    reconstruct a display trace by pairing each user turn with the assistant
    reply that follows it. System/tool messages carry no input/output pair and
    are skipped. A trailing user message with no reply yields outputs=None.
    """
    interactions: list[Interaction] = []
    pending_input: str | None = None
    for message in messages:
        if message.role == "user":
            if pending_input is not None:
                interactions.append(Interaction(inputs=pending_input, outputs=None))
            pending_input = message.content
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
    """Build and run a Giskard suite from a lidar scan. Filled in by later tasks."""

    async def _to_suite_result(self, scan_result, duration_ms: int) -> SuiteResult:
        scenario_results: list[ScenarioResult] = []
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
                        final_trace=await _trace_from_messages(attempt.messages),
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
