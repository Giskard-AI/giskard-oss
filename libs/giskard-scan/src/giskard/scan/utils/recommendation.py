"""Recommendation helpers for scan results."""

from giskard.checks import SuiteResult, get_default_generator

type QualitySummaryRow = dict[str, str | int | float | None]


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
        .with_inputs(
            component_results=_group_summary(result, "component"),
            quality_results=_group_summary(result, "quality"),
        )
        .run()
    )
    return (response.last.text or "").strip()


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
