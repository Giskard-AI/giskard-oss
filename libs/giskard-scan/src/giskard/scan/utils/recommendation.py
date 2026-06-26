"""Recommendation helpers for scan results."""

from typing import override

from giskard.checks import SuiteResult
from giskard.checks.core.result import GroupedSuiteResult, GroupStats
from giskard.checks.settings import get_default_generator
from pydantic import BaseModel, Field
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


class QualityRecommendationGeneration(BaseModel):
    """Structured output returned by the quality recommendation prompt."""

    recommendation: str = Field(
        default="",
        description="Concise recommendation text for improving quality scan failures.",
    )


type QualitySummaryRow = dict[str, str | int | float | None]


class QualityScanResult(SuiteResult, frozen=True):
    """Suite result enriched with a quality improvement recommendation.

    Attributes:
        recommendation: Markdown-friendly recommendation generated from the
            quality scan's component and quality-category failure profile.
    """

    recommendation: str = Field(default="")

    @override
    def group_by(self, key: str) -> "QualityGroupedSuiteResult":
        grouped = super().group_by(key)
        return QualityGroupedSuiteResult(
            suite_result=self,
            key=grouped.key,
            groups=grouped.groups,
        )

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield from SuiteResult.__rich_console__(self, console, options)
        recommendation = self.recommendation
        if recommendation.strip():
            yield _recommendation_panel(recommendation)


class QualityGroupedSuiteResult(GroupedSuiteResult, frozen=True):
    """Grouped quality scan result that appends the recommendation last."""

    suite_result: QualityScanResult
    groups: dict[str | None, GroupStats]

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield from SuiteResult.__rich_console__(self.suite_result, console, options)
        yield _group_table(self.key, self.groups)
        recommendation = self.suite_result.recommendation
        if recommendation.strip():
            yield _recommendation_panel(recommendation)


async def generate_quality_recommendation(result: SuiteResult) -> str:
    """Generate a recommendation for the failed quality scan scenarios.

    Args:
        result: Completed suite result from a quality scan.

    Returns:
        Recommendation text, or an empty string when the scan has no failures
        or errors to explain.
    """
    if not result.failures_and_errors:
        return ""

    response = (
        await get_default_generator()
        .template("giskard.scan::generate_suite/quality_recommendation.j2")
        .with_output(QualityRecommendationGeneration)
        .with_inputs(
            component_results=_group_summary(result, "component"),
            quality_results=_group_summary(result, "quality"),
        )
        .run()
    )
    return response.output.recommendation.strip()


def _recommendation_panel(recommendation: str) -> Panel:
    return Panel(
        Markdown(recommendation),
        title="Quality Recommendation",
        border_style="blue",
    )


def _group_table(key: str, groups: dict[str | None, GroupStats]) -> Table:
    table = Table(title=f"Results by {key}")
    table.add_column(key, style="bold")
    table.add_column("Pass Rate", justify="right")

    for group_value, stats in groups.items():
        if group_value is None:
            display_name = "(untagged)"
        elif group_value == "":
            display_name = "true"
        else:
            display_name = group_value
        rate = (
            f"{stats.passed} / {stats.non_skipped}"
            if stats.pass_rate is not None
            else "—"
        )
        table.add_row(display_name, rate)

    return table


def _group_summary(result: SuiteResult, key: str) -> list[QualitySummaryRow]:
    return [
        {
            "name": _display_group_name(group_name),
            "passed": stats.passed,
            "failed": stats.failed,
            "errored": stats.errored,
            "skipped": stats.skipped,
            "non_skipped": stats.non_skipped,
            "total": stats.total,
            "pass_rate": stats.pass_rate,
        }
        for group_name, stats in result.group_by(key).groups.items()
    ]


def _display_group_name(group_name: str | None) -> str:
    if group_name is None:
        return "(untagged)"
    if group_name == "":
        return "true"
    return group_name
