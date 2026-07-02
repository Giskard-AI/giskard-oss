"""Adapter that runs garak through Giskard scan scenarios."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterable
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, cast

from giskard.checks import (
    CheckResult,
    Metric,
    ScenarioResult,
    SuiteResult,
    Target,
    TestCaseResult,
    Trace,
)

from ...generators.base import TargetMode

if TYPE_CHECKING:
    from garak.attempt import Attempt
    from garak.detectors.base import Detector
    from garak.probes.base import Probe


logger = logging.getLogger(__name__)

# garak detector scores are in [0, 1]; a hit is anything above the midpoint.
# Strict ``>`` so an exactly-0.5 score is treated as a pass, matching garak's
# "uncertain, not a confirmed hit" reading of the boundary.
_HIT_THRESHOLD = 0.5


def garak_available() -> bool:
    """Return True if the optional garak dependency is importable."""
    return find_spec("garak") is not None


def _require_garak() -> None:
    if not garak_available():
        raise ImportError(
            "garak is not installed. Run: pip install giskard-scan[garak]"
        )


def _configure_garak() -> None:
    """Initialize garak's global config before any probe is loaded.

    garak reads ``_config.run.*`` at probe *instantiation* time (``Probe.__init__``
    copies ``generations``, ``seed``, ``parallel_attempts`` onto the instance), so
    this MUST run before ``_resolve_probes`` — setting it afterwards is a no-op on
    already-built probes. Without it, ``probe.probe()`` raises AttributeError on
    ``parallel_attempts`` / ``generations``.

    We pin ``generations = 1`` so a single prompt yields a single response: garak's
    default (5) fans out into 5 responses per prompt, which multiplies scenarios and,
    for multiturn (BFS) probes, pollutes the linear Trace. The report sink is a
    throwaway buffer — we assemble results from attempts, not from garak's report file.
    """
    import io

    import garak._config as garak_config

    garak_config.load_base_config()
    setattr(garak_config.run, "generations", 1)
    setattr(garak_config.transient, "reportfile", io.StringIO())


def _resolve_probes(probes: Iterable[str] | None) -> "list[Probe]":
    from garak._plugins import enumerate_plugins, load_plugin

    available = {
        full_name for full_name, loadable in enumerate_plugins("probes") if loadable
    }

    if probes is not None:
        requested = set(probes)
        missing = requested - available
        if missing:
            logger.warning(f"Unknown probes: {missing}")
        probe_names = requested & available
    else:
        probe_names = available

    loaded_probes: list[Probe] = []
    for probe_name in probe_names:
        try:
            loaded_probes.append(cast("Probe", load_plugin(probe_name)))
        except Exception as exc:  # noqa: BLE001 — one bad plugin must not abort the scan
            logger.warning("Failed to load probe %s: %s", probe_name, exc)

    if probes is not None:
        return loaded_probes

    # Filter out inactive probes if a list of probes was not provided
    return [probe for probe in loaded_probes if probe.active]


def _resolve_detectors(probe: "Probe") -> "list[Detector]":
    from garak._plugins import load_plugin

    detectors = [probe.primary_detector or "always.Fail"] + list(
        probe.extended_detectors
    )
    return [
        cast("Detector", load_plugin(f"detectors.{detector}")) for detector in detectors
    ]


class GarakScanAdapter:
    """Build and run a Giskard suite from garak probes."""

    def _evaluate_attempt(
        self, attempt: "Attempt", detectors: "list[Detector]"
    ) -> list[TestCaseResult]:
        check_results = defaultdict(list)

        for detector in detectors:
            try:
                # detect() is typed as Iterable; materialize so we can measure it
                # and index safely (a one-shot generator would be exhausted below).
                scores = list(detector.detect(attempt))

                conversation_count = len(attempt.conversations)
                if len(scores) > conversation_count:
                    # garak detectors are contracted to return one score per
                    # conversation. Extra scores index past the assembly loop below
                    # and would be dropped silently, so surface the misbehaving
                    # detector instead of hiding the lost results.
                    logger.warning(
                        "Detector %s returned %d scores for %d conversations; "
                        "extra scores are ignored",
                        detector.name,
                        len(scores),
                        conversation_count,
                    )

                for conversation_idx, score in enumerate(scores):
                    if score is None:
                        check_results[conversation_idx].append(
                            CheckResult.skip(
                                message="detector returned no score",
                                details={"detector": detector.name},
                            )
                        )
                    elif score > _HIT_THRESHOLD:
                        check_results[conversation_idx].append(
                            CheckResult.failure(
                                message=detector.hit_desc,
                                details={"detector": detector.name},
                                metrics=[Metric(name=str(detector.name), value=score)],
                            )
                        )
                    else:
                        check_results[conversation_idx].append(
                            CheckResult.success(
                                message=detector.pass_desc,
                                details={"detector": detector.name},
                                metrics=[Metric(name=str(detector.name), value=score)],
                            )
                        )
            except Exception as exc:  # noqa: BLE001 — a broken detector skips, not aborts
                for conversation_idx in range(len(attempt.conversations)):
                    check_results[conversation_idx].append(
                        CheckResult.error(
                            message="detector raised",
                            details={"detector": detector.name, "error": repr(exc)},
                        )
                    )

        test_case_results: list[TestCaseResult] = []
        for conversation_idx in range(len(attempt.conversations)):
            results = check_results[conversation_idx]
            if not results:
                results = [
                    CheckResult.skip(
                        message="no detector score for this conversation",
                        details={"conversation_idx": conversation_idx},
                    )
                ]
            test_case_results.append(TestCaseResult(results=results, duration_ms=0))
        return test_case_results

    def _run_probe[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        probe: "Probe",
        target: Target[InputType, OutputType, TraceType],
        loop: asyncio.AbstractEventLoop,
    ) -> list[ScenarioResult[TraceType]]:
        from ._generator import TargetGenerator

        generator = TargetGenerator(target=target, loop=loop)
        detectors = _resolve_detectors(probe)

        attempts = probe.probe(generator)
        scenario_results = []
        for attempt_idx, attempt in enumerate(attempts):
            test_case_results = self._evaluate_attempt(attempt, detectors)
            for conversation_idx, test_case_result in enumerate(test_case_results):
                scenario_results.append(
                    ScenarioResult(
                        scenario_name=f"Garak {probe.probename} #{attempt_idx + 1} — run {conversation_idx + 1}",
                        steps=[test_case_result],
                        final_trace=generator.get_trace(
                            attempt.conversations[conversation_idx]
                        ),
                        duration_ms=0,
                    )
                )

        return scenario_results

    def _run_probe_isolated[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        probe: "Probe",
        target: Target[InputType, OutputType, TraceType],
        loop: asyncio.AbstractEventLoop,
    ) -> list[ScenarioResult[TraceType]]:
        try:
            return self._run_probe(probe, target, loop)
        except Exception as exc:  # noqa: BLE001 — probe errors are ignored per spec
            logger.warning("Probe %s raised: %s", probe.probename, exc)
            return []

    async def run[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
        self,
        target: Target[InputType, OutputType, TraceType],
        **kwargs: Any,
    ) -> SuiteResult:
        _require_garak()
        from garak.probes.base import IterativeProbe

        # Must run before _resolve_probes: garak binds run config onto probes at
        # instantiation time (see _configure_garak).
        _configure_garak()

        probes = _resolve_probes(kwargs.pop("probes", None))
        target_mode: TargetMode = kwargs.pop("target_mode", "multiturn")

        if target_mode == "singleturn":
            probes = [
                probe for probe in probes if not isinstance(probe, IterativeProbe)
            ]

        loop = asyncio.get_running_loop()
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(
                    asyncio.to_thread(self._run_probe_isolated, probe, target, loop)
                )
                for probe in probes
            ]

        # Results are only available once the TaskGroup has exited and every
        # task has completed; reading task.result() inside the block would hit
        # InvalidStateError because the tasks have not run yet.
        scenario_results = [scenario for task in tasks for scenario in task.result()]

        return SuiteResult(results=scenario_results, duration_ms=0)
