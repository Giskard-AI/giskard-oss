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

from .._shared import reject_unexpected_kwargs, trace_from_role_content_turns

logger = logging.getLogger(__name__)

# Lidar tags its multiturn probes (crescendo, goat, ...) with this. It is the
# ONLY uniform signal lidar exposes for "this probe drives a multi-turn
# conversation" — there is no shared base class or attribute to isinstance on
# (unlike garak's IterativeProbe). We use it to drop those probes when the
# caller declares a single-turn target. A guard test pins this string so a lidar
# rename fails loudly instead of silently leaking multiturn probes into
# singleturn mode. See test_lidar_singleturn.py.
_MULTITURN_TAG = "gsk:probe-type='multi-turn'"


def _singleturn_probe_ids(
    probes: "list[str] | None", tags: "list[str] | None"
) -> list[str]:
    """Resolve the probe ids to run in single-turn mode.

    Lidar's ``tags_filter`` is inclusion-only (a probe is kept if it matches
    ANY tag), so it cannot express "exclude multiturn". Instead we resolve the
    exact class set lidar would run for the caller's ``probes``/``tags`` filters,
    then drop every probe carrying ``_MULTITURN_TAG``. The surviving ids are
    passed to ``run_scan`` with ``tags_filter=None`` (the tag filter has already
    been applied here).
    """
    from lidar.utils.probe_registry import ProbeRegistry

    candidates = ProbeRegistry().get_probes(tags, probe_ids=probes)
    return [
        info.id
        for probe_cls in candidates
        if _MULTITURN_TAG not in (info := probe_cls.info()).tags
    ]


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

    Lidar owns its own executor and hands back finished conversations, so there
    is no typed Trace to recover and we pair the turns instead.
    """
    return await trace_from_role_content_turns(messages)


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
        # target_mode is a scan-wide hint about the target's conversational
        # ability. Lidar owns its own turn logic, so the only thing "singleturn"
        # can mean here is: don't run probes that require a multi-turn exchange
        # (crescendo, goat, ...). Those probes have no single-turn form — a
        # 1-turn crescendo is not a valid attack — so we SKIP them, mirroring
        # garak's IterativeProbe skip. "multiturn" (the default) runs everything.
        target_mode = kwargs.pop("target_mode", None)
        reject_unexpected_kwargs("lidar", kwargs)

        # tags_filter is applied by run_scan for the default path; in singleturn
        # mode we resolve+filter the probe ids ourselves and pass tags_filter=None
        # so lidar doesn't intersect the tag filter a second time.
        tags_filter: list[str] | None = tags
        if target_mode == "singleturn":
            probes = _singleturn_probe_ids(probes, tags)
            tags_filter = None
            if not probes:
                logger.debug("target_mode='singleturn' left no lidar probes to run.")
                return SuiteResult(results=[], duration_ms=0)

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
            target=bridged,
            target_info=target_info,
            probe_ids=probes,
            tags_filter=tags_filter,
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
