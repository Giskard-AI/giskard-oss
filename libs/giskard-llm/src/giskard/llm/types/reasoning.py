"""Reasoning: effort levels and the ``ReasoningContent`` carrier (ADR-0003)."""

from collections.abc import Iterable
from typing import Annotated, Literal, Required, TypedDict, cast, get_args

from pydantic import BeforeValidator, Field

from ._base import _BaseModel

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


# -- Reasoning content (ADR-0003) ----------------------------------------------------

ReasoningStatus = Literal["in_progress", "completed", "incomplete"]


class ReasoningSummary(_BaseModel):
    """``summary_text`` part of a reasoning block."""

    type: Literal["summary_text"] = "summary_text"
    text: str


class ReasoningText(_BaseModel):
    """``reasoning_text`` part of a reasoning block."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ReasoningContent(_BaseModel):
    """One provider reasoning block, as a chat content part or a Responses item.

    Shape of the OpenAI Responses ``reasoning`` item plus ``signature`` and
    ``provider``. One provider block maps to exactly one ``ReasoningContent``.

    - ``signature`` authenticates the visible ``content`` (Anthropic
      ``thinking.signature``, base64 Gemini thought signature). An empty
      ``content`` text with a signature is still a block to replay.
    - ``encrypted_content`` replaces content the caller cannot see (OpenAI
      encrypted reasoning, Anthropic ``redacted_thinking.data``).
    - ``provider`` is the Originating Provider: the giskard provider id the call
      was routed through. Serializers replay a part only to that exact provider
      and never send this field. ``provider=None`` is never replayed.
    - ``id`` is never invented; only OpenAI-tagged reasoning carries one.

    Never contributes to ``text``, ``transcript`` or ``output_text``.
    """

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    summary: list[ReasoningSummary] = Field(default_factory=list)
    content: list[ReasoningText] | None = None
    encrypted_content: str | None = None
    signature: str | None = None
    status: ReasoningStatus | None = None
    provider: str | None = None


def without_reasoning[T](parts: Iterable[T]) -> list[T] | None:
    """Drop ``ReasoningContent`` parts, keeping every other part in order.

    For callers that show or send a conversation without its private reasoning.
    A list left empty becomes ``None``, so it reads as "no content".
    """
    kept = [part for part in parts if not isinstance(part, ReasoningContent)]
    return kept or None


class ReasoningSummaryParam(TypedDict, total=False):
    type: Required[Literal["summary_text"]]
    text: Required[str]


class ReasoningTextParam(TypedDict, total=False):
    type: Required[Literal["reasoning_text"]]
    text: Required[str]


class ReasoningContentParam(TypedDict, total=False):
    type: Required[Literal["reasoning"]]
    id: str | None
    summary: list[ReasoningSummaryParam]
    content: list[ReasoningTextParam] | None
    encrypted_content: str | None
    signature: str | None
    status: ReasoningStatus | None
    provider: str | None
