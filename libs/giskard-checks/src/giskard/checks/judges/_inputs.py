"""Shared input resolution helpers for LLM judges."""

from typing import Any, NamedTuple

from ..core.extraction import NoMatch, provided_or_resolve
from ..core.interaction import Trace
from ..core.result import CheckResult


class ResolvableInput(NamedTuple):
    """A judge input that may be supplied directly or extracted from the trace.

    Attributes
    ----------
    name : str
        Field name used to build the error message and details (e.g. ``"answer"``,
        reported as ``"No value found for answer key '...'."``).
    key : str
        JSONPath expression used when ``value`` is ``MISSING``.
    value : Any
        Directly-provided value, which takes priority over ``key``.
    """

    name: str
    key: str
    value: Any


def error_if_unresolved[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    trace: TraceType,
    *inputs: ResolvableInput,
) -> CheckResult | None:
    """Return ERROR for the first input whose extraction yields ``NoMatch``.

    Inputs are checked in the order given, so callers should list the most
    fundamental one first: on a fully missing trace every key misses, and the
    first reported is the one most likely to explain the misconfiguration.

    Returning ERROR (rather than FAIL) keeps an unresolvable key distinct from
    a genuine judge verdict: ``Not(...)`` inverts PASS/FAIL but leaves ERROR
    untouched, so a broken key cannot be laundered into a green result.

    Returns ``None`` when every input resolves (including to empty strings).
    """
    for name, key, value in inputs:
        resolved = provided_or_resolve(trace, key=key, value=value)
        if isinstance(resolved, NoMatch):
            return CheckResult.error(
                message=f"No value found for {name} key '{key}'.",
                details={f"{name}_key": key, name: resolved},
            )

    return None


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
    return error_if_unresolved(
        trace,
        ResolvableInput("answer", answer_key, answer),
        ResolvableInput("context", context_key, context),
    )
