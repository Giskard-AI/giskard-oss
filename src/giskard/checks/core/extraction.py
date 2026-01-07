from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast, override

from giskard.core import Discriminated, discriminated_base
from jsonpath_ng import parse
from pydantic import Field

from .trace import Trace


@discriminated_base
class Extractor[InputType, OutputType](Discriminated, ABC):
    """Base extractor strategy.

    Implementations return a list of extracted items from a `Trace`.
    """

    @abstractmethod
    def extract(self, trace: Trace[InputType, OutputType]) -> list[Any]:
        """Extract values from the trace.

        Parameters
        ----------
        trace : Trace
            The trace to extract values from.

        Returns
        -------
        list[Any]
            List of extracted values.
        """
        raise NotImplementedError


@Extractor.register("jsonpath")
class JsonPathExtractor[InputType, OutputType](Extractor[InputType, OutputType]):
    """Extract values using a JSONPath expression evaluated on `{"trace": trace.model_dump()}`.

    The extractor unwraps objects that expose a `value` attribute (e.g., jsonpath-ng DatumInContext)
    and returns a plain list of Python values.

    Note
    ----
    JSONPath extraction operates on dictionaries, not objects, so it uses `trace.model_dump()`.
    This means that `@property` methods are not accessible unless they are decorated with
    `@computed_field` from Pydantic, which includes them in `model_dump()`.

    Examples
    --------
    To make a custom property available to JSONPath extraction, use `@computed_field`::

        from pydantic import computed_field

        class MyTrace(Trace[str, str]):
            @computed_field
            @property
            def messages(self) -> list[str]:
                return [i.outputs for i in self.interactions]

    Then JSONPath expressions like `trace.messages` will work.
    """

    key: str = Field(
        ...,
        description="JSONPath expression to extract values (e.g., 'trace.interactions[-1].outputs')",
    )

    @override
    def extract(self, trace: Trace[InputType, OutputType]) -> list[Any]:
        """Extract values using JSONPath.

        Parameters
        ----------
        trace : Trace
            The trace to extract values from.

        Returns
        -------
        list[Any]
            List of extracted values.
        """
        expr = parse(self.key)
        matches = expr.find({"trace": trace.model_dump()})
        results: list[Any] = []
        for item in matches:
            value = getattr(item, "value", item)
            results.append(value)
        return results


def resolve[InputType, OutputType](
    trace: Trace[InputType, OutputType], key: str, multiple: bool = False
) -> list[Any] | Any | None:
    """Resolve a JSONPath expression against a trace.

    Parameters
    ----------
    trace : Trace
        The trace to resolve against.
    key : str
        JSONPath expression to evaluate (e.g., 'trace.interactions[-1].outputs').
    multiple : bool
        If True, return a list of all matches. If False, return the first match or None.

    Returns
    -------
    list[Any] | Any | None
        The resolved value(s) or None if no matches found.

    Note
    ----
    JSONPath extraction operates on dictionaries, not objects, so it uses `trace.model_dump()`.
    This means that `@property` methods are not accessible unless they are decorated with
    `@computed_field` from Pydantic, which includes them in `model_dump()`.
    """
    expr = parse(key)
    matches = expr.find({"trace": trace.model_dump()})

    if not matches:
        return [] if multiple else None

    values = [m.value for m in matches]

    return values if multiple else values[0]


def provided_or_resolve[ValueType, InputType, OutputType](
    value: ValueType | None,
    trace: Trace[InputType, OutputType],
    key: str,
    multiple: bool = False,
) -> ValueType | None:
    """Resolve a JSONPath expression against a trace.

    Parameters
    ----------
    trace : Trace
    key : str
    multiple : bool
    """
    if value is not None:
        return value

    return cast(ValueType, resolve(trace, key, multiple))
