"""Hub format export for SuiteResult."""

from typing import Any

from ..core.result import SuiteResult


def to_hub_format(result: SuiteResult) -> dict[str, Any]:
    """Convert a SuiteResult into a JSON-serializable Giskard Hub payload.

    The returned dict is the payload accepted by the Giskard Hub
    ``POST /v2/evaluations/upload`` endpoint and can be passed directly to
    :meth:`giskard_hub.HubClient.evaluations.upload`.

    Parameters
    ----------
    result : SuiteResult
        The suite result to convert.

    Returns
    -------
    dict[str, Any]
        JSON-serializable representation of the suite result, containing
        ``results`` (list of scenario results), ``duration_ms``, and aggregate
        computed fields (``passed_count``, ``failed_count``, ``errored_count``,
        ``skipped_count``, ``pass_rate``).
    """
    return result.model_dump(mode="json")
