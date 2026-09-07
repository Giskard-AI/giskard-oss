"""Hub format export for SuiteResult."""

from typing import Any

from giskard.core import scoped_telemetry, telemetry_capture, telemetry_tag

from ..core.result import SuiteResult


@scoped_telemetry
def to_hub_format(result: SuiteResult) -> dict[str, Any]:
    """Convert a SuiteResult into a JSON-serializable Giskard Hub payload.

    The returned dict is the payload accepted by the Giskard Hub API and the Giskard Hub Python SDK.

    Parameters
    ----------
    result : SuiteResult
        The suite result to convert.

    Returns
    -------
    dict[str, Any]
        JSON-serializable representation of the suite result
    """
    telemetry_tag("giskard_component", "export")
    telemetry_tag("giskard_operation", "to_hub_format")
    payload = result.model_dump(mode="json", fallback=str)
    telemetry_capture(
        "checks_hub_exported",
        properties={
            "integration": "giskard-checks",
            "scenario_count": len(result.results),
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "errored_count": result.errored_count,
            "skipped_count": result.skipped_count,
            "has_recommendation": bool(result.recommendation),
        },
    )
    return payload
