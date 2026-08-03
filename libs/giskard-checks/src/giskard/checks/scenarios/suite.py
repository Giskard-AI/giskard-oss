import asyncio
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar

from giskard.core import telemetry_capture, telemetry_run_context, telemetry_tag
from pydantic import BaseModel, Field
from pydantic.experimental.missing_sentinel import MISSING
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from .._telemetry_props import suite_shape_properties
from ..core.interaction import Trace
from ..core.result import (
    STATUS_SUMMARY_ORDER,
    ScenarioResult,
    ScenarioStatus,
    SuiteResult,
    format_status_count_text,
)
from ..core.scenario import Scenario
from ..core.types import Target
from ..utils.checkpoint import (
    RunStore,
    ensure_checkpoint_id,
    resolve_checkpoint_options,
    run_fingerprint,
)

type ResumeMode = bool | Literal["force"]

InputType = TypeVar("InputType", infer_variance=True)
OutputType = TypeVar("OutputType", infer_variance=True)


class _OverallOnly(ProgressColumn):
    """Render the wrapped column only for the overall task, not per-scenario rows."""

    def __init__(self, column: ProgressColumn) -> None:
        super().__init__()
        self._column: ProgressColumn = column

    def render(self, task: "Task") -> RenderableType:
        if not task.fields.get("overall"):
            return Text("")
        return self._column.render(task)


class _SuiteProgress(Progress):
    """Progress display that appends a live status-count summary below the rows.

    As scenarios finish, ``record`` tallies their outcomes and the overridden
    ``get_renderables`` draws a colored ``errored, failed, skipped, passed`` line
    under the task rows, mirroring the final suite summary.
    """

    def __init__(self, *columns: ProgressColumn, disable: bool = False) -> None:
        # Set before super().__init__(): it renders get_renderables(), which reads _counts.
        self._counts: dict[str, int] = {key: 0 for key, _ in STATUS_SUMMARY_ORDER}
        self._overall_id: TaskID | None = None
        super().__init__(*columns, disable=disable)

    def start_overall(self, total: int) -> None:
        self._overall_id = self.add_task("Running scenarios", total=total, overall=True)

    def describe(self, text: str) -> None:
        if self._overall_id is not None:
            self.update(self._overall_id, description=text)

    def record(self, status: ScenarioStatus) -> None:
        """Tally one finished scenario and advance the overall bar."""
        self._counts[status.value] += 1
        if self._overall_id is not None:
            self.advance(self._overall_id)

    @contextmanager
    def scenario_row(self, name: str) -> Iterator[None]:
        """Show a row for one scenario while it runs, then remove it."""
        task_id = self.add_task(f"  ↳ {name}", total=None)
        try:
            yield
        finally:
            self.remove_task(task_id)

    def _summary_renderable(self) -> Text | None:
        """Colored counts line, or ``None`` until at least one scenario has finished."""
        return format_status_count_text(self._counts, prefix="  ")

    def get_renderables(self) -> Iterable[RenderableType]:
        yield self.make_tasks_table(self.tasks)
        summary = self._summary_renderable()
        if summary is not None:
            yield summary


class Suite(BaseModel, Generic[InputType, OutputType]):
    """A suite of scenarios that can be run together with a shared target.

    A suite holds multiple scenarios and can run them serially, optionally
    binding a single target SUT to all scenarios at once. If a target is
    provided at the suite level or during the run call, it overrides any
    scenario-level target.

    Attributes
    ----------
    name : str
        Suite identifier.
    scenarios : list[Scenario]
        List of scenarios to execute.
    target : Target | MISSING
        Optional suite-level target SUT.

    Examples
    --------
    ```python
    from giskard.checks import Suite, Scenario

    scenario1 = Scenario("scenario_1").interact("hello")
    scenario2 = Scenario("scenario_2").interact("hi")

    suite = Suite(name="my_suite", target=my_sut)
    suite.append(scenario1).append(scenario2)

    result = await suite.run()
    print(result.pass_rate)
    ```
    """

    name: str = Field(..., description="Suite name")
    scenarios: list[Scenario[InputType, OutputType, Trace[Any, Any]]] = Field(
        default_factory=list, description="Scenarios in the suite"
    )
    target: Target[InputType, OutputType, Trace[Any, Any]] | MISSING = Field(  # pyright: ignore[reportInvalidTypeForm]
        default=MISSING,
        description="Suite-level target SUT that will override any scenario-level target.",
    )

    def append(
        self,
        scenario: Scenario[InputType, OutputType, Trace[Any, Any]],
    ) -> Self:
        """Add a scenario to the suite.

        Parameters
        ----------
        scenario : Scenario
            The scenario to add to the suite.

        Returns
        -------
        Suite
            The suite itself, allowing fluent chaining.
        """
        self.scenarios.append(scenario)
        return self

    async def run(
        self,
        target: Target[InputType, OutputType, Trace[Any, Any]] | MISSING = MISSING,  # pyright: ignore[reportInvalidTypeForm]
        return_exception: bool = False,
        parallel: bool = False,
        max_concurrency: int | None = None,
        verbose: bool = True,
        checkpoint_dir: Path | str | bool | None = None,
        resume: ResumeMode | None = None,
    ) -> SuiteResult:
        """Run all scenarios in the suite.

        Parameters
        ----------
        target : Target | MISSING, optional
            Override target for all scenarios in the suite. If provided, this
            overrides both the suite-level target and any scenario-level targets.
        return_exception : bool
            If True, return results even when exceptions occur instead of raising.
        parallel : bool
            If True, run all scenarios concurrently while preserving result order.
        max_concurrency : int | None
            Max concurrent scenarios when ``parallel=True`` (positive int).
            ``None`` (default) is unbounded: all scenarios start at once, so the
            provider's rate limits become the effective cap.
        verbose : bool
            If True (default), display a progress bar showing which scenario is
            currently running. Set to False for non-interactive environments.
        checkpoint_dir : Path | str | bool | None, optional
            Checkpoint root. ``None`` (default) uses
            ``.giskard/checkpoints/<fingerprint>/`` (or ``GISKARD_CHECKPOINT_DIR``
            as root). ``False`` disables checkpointing. A path is used as the
            root with a fingerprint subdir. Checkpoints may contain sensitive
            traces; keep them local.
        resume : bool | Literal["force"] | None, optional
            Skip scenarios already present in the checkpoint. Defaults to
            ``True`` when checkpointing is enabled. Use ``False`` for a fresh
            store, or ``"force"`` to ignore fingerprint mismatches.

        Returns
        -------
        SuiteResult
            Aggregated results from all scenarios.

        Examples
        --------
        ```python
        from giskard.checks import Suite

        suite = Suite(name="my_suite", target=my_sut_v1)
        suite.append(scenario_1).append(scenario_2)
        result_v1 = await suite.run()
        result_v2 = await suite.run(target=my_sut_v2)
        ```
        """
        target = target if target is not MISSING else self.target
        has_target = target is not MISSING

        if max_concurrency is not None:
            if not isinstance(max_concurrency, int) or isinstance(
                max_concurrency, bool
            ):
                raise TypeError("max_concurrency must be None or a positive integer")
            if max_concurrency < 1:
                raise ValueError("max_concurrency must be greater than 0")

        scenario_ids = [
            ensure_checkpoint_id(
                scenario.annotations,
                name=scenario.name,
                index=index,
                suite_name=self.name,
            )
            for index, scenario in enumerate(self.scenarios)
        ]
        fingerprint = run_fingerprint(self.name, scenario_ids)
        ck_path, resume_mode = resolve_checkpoint_options(
            checkpoint_dir, resume, fingerprint=fingerprint
        )
        store: RunStore | None = None
        cached: dict[str, ScenarioResult[Trace[Any, Any]]] = {}
        if ck_path is not None:
            store = await RunStore.open(
                ck_path, fingerprint=fingerprint, resume=resume_mode
            )
            if resume_mode:
                for cid, payload in store.load_payloads("scenario_finished").items():
                    cached[cid] = ScenarioResult.model_validate(payload)

        with telemetry_run_context():
            telemetry_tag("giskard_component", "suite")
            telemetry_tag("giskard_operation", "suite_run")

            shape_props = suite_shape_properties(
                scenario_count=len(self.scenarios),
                has_target=has_target,
                parallel=parallel,
            )
            telemetry_capture(
                "checks_suite_run_started",
                properties=shape_props,
            )

            start_time = time.perf_counter()
            with self._progress_bar(enabled=verbose) as tracker:
                if parallel:
                    results = await self._run_parallel(
                        target,
                        return_exception,
                        max_concurrency,
                        tracker,
                        scenario_ids=scenario_ids,
                        store=store,
                        cached=cached,
                    )
                else:
                    results = await self._run_serial(
                        target,
                        return_exception,
                        tracker,
                        scenario_ids=scenario_ids,
                        store=store,
                        cached=cached,
                    )
            end_time = time.perf_counter()

            suite_result = SuiteResult(
                results=results,
                duration_ms=int((end_time - start_time) * 1000),
                suite=self,
            )

            telemetry_capture(
                "checks_suite_run_finished",
                properties={
                    **shape_props,
                    "duration_ms": suite_result.duration_ms,
                    "passed_count": suite_result.passed_count,
                    "failed_count": suite_result.failed_count,
                    "errored_count": suite_result.errored_count,
                    "skipped_count": suite_result.skipped_count,
                },
            )

        return suite_result

    @contextmanager
    def _progress_bar(self, enabled: bool) -> Iterator[_SuiteProgress]:
        """Yield a progress tracker; a disabled no-op display when ``enabled`` is False."""
        with _SuiteProgress(
            _OverallOnly(SpinnerColumn()),
            TextColumn("[progress.description]{task.description}"),
            _OverallOnly(BarColumn()),
            _OverallOnly(MofNCompleteColumn()),
            _OverallOnly(TimeElapsedColumn()),
            disable=not enabled,
        ) as progress:
            progress.start_overall(len(self.scenarios))
            yield progress

    async def _run_one(
        self,
        scenario: Scenario[InputType, OutputType, Trace[Any, Any]],
        checkpoint_id: str,
        target: Any,
        return_exception: bool,
        store: RunStore | None,
        cached: dict[str, ScenarioResult[Trace[Any, Any]]],
    ) -> ScenarioResult[Trace[Any, Any]]:
        if checkpoint_id in cached:
            return cached[checkpoint_id]
        result = await scenario.run(target=target, return_exception=return_exception)
        if store is not None:
            await store.append(
                "scenario_finished",
                id=checkpoint_id,
                payload=result.model_dump(mode="json"),
            )
            cached[checkpoint_id] = result
        return result

    async def _run_serial(
        self,
        target: Any,
        return_exception: bool,
        progress: _SuiteProgress,
        *,
        scenario_ids: list[str],
        store: RunStore | None,
        cached: dict[str, ScenarioResult[Trace[Any, Any]]],
    ) -> list[ScenarioResult[Trace[Any, Any]]]:
        results: list[ScenarioResult[Trace[Any, Any]]] = []
        for scenario, checkpoint_id in zip(self.scenarios, scenario_ids, strict=True):
            progress.describe(f"Running: {scenario.name}")
            result = await self._run_one(
                scenario, checkpoint_id, target, return_exception, store, cached
            )
            results.append(result)
            progress.record(result.status)
        return results

    async def _run_parallel(
        self,
        target: Any,
        return_exception: bool,
        max_concurrency: int | None,
        progress: _SuiteProgress,
        *,
        scenario_ids: list[str],
        store: RunStore | None,
        cached: dict[str, ScenarioResult[Trace[Any, Any]]],
    ) -> list[ScenarioResult[Trace[Any, Any]]]:
        semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else nullcontext()
        )

        async def run_scenario(
            scenario: Scenario[InputType, OutputType, Trace[Any, Any]],
            checkpoint_id: str,
        ) -> ScenarioResult[Trace[Any, Any]]:
            async with semaphore:
                with progress.scenario_row(scenario.name):
                    result = await self._run_one(
                        scenario,
                        checkpoint_id,
                        target,
                        return_exception,
                        store,
                        cached,
                    )
                progress.record(result.status)
            return result

        try:
            async with asyncio.TaskGroup() as task_group:
                tasks = [
                    task_group.create_task(run_scenario(scenario, checkpoint_id))
                    for scenario, checkpoint_id in zip(
                        self.scenarios, scenario_ids, strict=True
                    )
                ]
        except* Exception as exc_group:
            if len(exc_group.exceptions) == 1:
                raise exc_group.exceptions[0]
            raise
        return [task.result() for task in tasks]


# `SuiteResult.suite` is a forward reference to `Suite`, which is only imported
# under `TYPE_CHECKING` in result.py to avoid a circular import. Rebuild the model
# here, where `Suite` exists at runtime, so Pydantic can resolve the annotation.
SuiteResult.model_rebuild()
