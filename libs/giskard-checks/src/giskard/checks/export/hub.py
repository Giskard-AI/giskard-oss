"""Hub format export for SuiteResult."""

from typing import Any

from ..core.result import SuiteResult


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
    return result.model_dump(mode="json", fallback=str)
