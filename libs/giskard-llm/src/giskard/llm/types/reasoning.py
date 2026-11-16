"""Canonical reasoning effort levels and Pydantic case-normalization helpers."""

from typing import Annotated, Literal, cast, get_args

from pydantic import BeforeValidator

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORT_LEVELS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)


def normalize_reasoning_effort(value: object) -> ReasoningEffort:
    """Case-normalize and validate a canonical reasoning effort level.

    Parameters
    ----------
    value : object
        Candidate reasoning effort value. Must be a ``str`` matching one of
        :data:`REASONING_EFFORT_LEVELS` (case-insensitive).

    Returns
    -------
    ReasoningEffort
        Lowercased canonical effort level.

    Raises
    ------
    ValueError
        If *value* is not a string or is outside the seven canonical levels.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"reasoning_effort must be a string, got {type(value).__name__}"
        )
    normalized = value.lower()
    if normalized not in REASONING_EFFORT_LEVELS:
        allowed = ", ".join(REASONING_EFFORT_LEVELS)
        raise ValueError(
            f"Invalid reasoning_effort {value!r}; expected one of: {allowed}"
        )
    return cast(ReasoningEffort, normalized)


NormalizedReasoningEffort = Annotated[
    ReasoningEffort, BeforeValidator(normalize_reasoning_effort)
]
