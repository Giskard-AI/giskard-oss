"""Shared input resolution helpers for answer/context LLM judges."""

from typing import Any

from ..core.extraction import NoMatch, provided_or_resolve
from ..core.interaction import Trace
from ..core.result import CheckResult


def error_if_unresolved_answer_or_context[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    trace: TraceType,
    *,
    answer: Any,
    answer_key: str,
    context: Any,
    context_key: str,
) -> CheckResult | None:
    """Return ERROR when answer or context extraction yields ``NoMatch``.

    Checks answer before context so a fully missing trace reports the answer
    key. Returns ``None`` when both inputs resolve (including empty strings).
    """
    resolved_answer = provided_or_resolve(trace, key=answer_key, value=answer)
    if isinstance(resolved_answer, NoMatch):
        return CheckResult.error(
            message=f"No value found for answer key '{answer_key}'.",
            details={"answer_key": answer_key, "answer": resolved_answer},
        )

    resolved_context = provided_or_resolve(trace, key=context_key, value=context)
    if isinstance(resolved_context, NoMatch):
        return CheckResult.error(
            message=f"No value found for context key '{context_key}'.",
            details={"context_key": context_key, "context": resolved_context},
        )

    return None
