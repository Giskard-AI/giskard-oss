from giskard.checks.core.result import SuiteResult
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def test_suite_result_renders_recommendation_after_group_table():
    result = SuiteResult(
        results=[],
        duration_ms=0,
        recommendation="- Improve LLM answers for conciseness failures.",
    )
    console = Console(width=100)

    suite_renderables = list(result.__rich_console__(console, console.options))
    grouped_renderables = list(
        result.group_by("component").__rich_console__(console, console.options)
    )

    assert any(
        isinstance(renderable, Panel) and renderable.title == "Recommendation"
        for renderable in suite_renderables
    )
    group_table_index = next(
        index
        for index, renderable in enumerate(grouped_renderables)
        if isinstance(renderable, Table) and renderable.title == "Results by component"
    )
    recommendation_index = next(
        index
        for index, renderable in enumerate(grouped_renderables)
        if isinstance(renderable, Panel) and renderable.title == "Recommendation"
    )
    assert group_table_index < recommendation_index
