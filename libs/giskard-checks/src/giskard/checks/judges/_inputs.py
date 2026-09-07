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
        Human-readable label used to build the error message (e.g. ``"answer"``,
        reported as ``"No value found for answer key '...'."``) and to name the
        resolved value in ``details``.
    key : str
        JSONPath expression used when ``value`` is ``MISSING``.
    value : Any
        Directly-provided value, which takes priority over ``key``.
    key_field : str | None
        Name of the model field holding ``key``, used as the ``details`` entry
        so users can act on the reported key. Defaults to ``f"{name}_key"``,
        which is correct for every input whose field follows that convention;
        the subject input passes ``"target_key"`` explicitly, since its domain
        label ("answer", "output") is only a read alias there.
    """

    name: str
    key: str
    value: Any
    key_field: str | None = None


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
    for name, key, value, key_field in inputs:
        resolved = provided_or_resolve(trace, key=key, value=value)
        if isinstance(resolved, NoMatch):
            return CheckResult.error(
                message=f"No value found for {name} key '{key}'.",
                details={key_field or f"{name}_key": key, name: resolved},
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
        # The answer is the subject under test, so its field is ``target_key``
        # even though the judge-facing label stays "answer".
        ResolvableInput("answer", answer_key, answer, "target_key"),
        ResolvableInput("context", context_key, context),
    )
