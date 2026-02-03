from __future__ import annotations

from typing import Any, override

from giskard.core import NOT_PROVIDED, NotProvided
from jsonpath_ng import (
    Child,
    DatumInContext,
    Descendants,
    Intersect,
    JSONPath,
    Slice,
    Union,
    Where,
    WhereNot,
    parse,
)
from pydantic import BaseModel, Field

from .trace import Trace


class NoMatch(BaseModel):
    """Indicates that a key was provided but no match was found during extraction.

    This class is returned by the `provided` function when a JSONPath expression
    doesn't match any value in the data structure.
    """

    key: str = Field(
        ..., description="The key that was provided but no match was found"
    )

    @override
    def __str__(self) -> str:
        return f"No match for key: {self.key}"

    @override
    def __repr__(self) -> str:
        return f"NoMatch(key={self.key})"


def _is_list_expression(expression: JSONPath) -> bool:
    if isinstance(expression, Child | Descendants):
        return _is_list_expression(expression.right) or _is_list_expression(
            expression.left
        )

    if isinstance(expression, Where | WhereNot):
        return _is_list_expression(expression.left)

    if isinstance(expression, Slice | Union | Intersect):
        return True

    return False


def resolve[TraceType: Trace](trace: TraceType, key: str) -> Any:  # pyright: ignore[reportMissingTypeArgument]
    expression: JSONPath = parse(key)
    matches: list[DatumInContext] = expression.find({"trace": trace.model_dump()})

    if len(matches) > 1 or _is_list_expression(expression):
        return [m.value for m in matches]

    return matches[0].value if matches else NoMatch(key=key)


def provided_or_resolve[TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    trace: TraceType,
    key: str | NotProvided = NOT_PROVIDED,
    value: Any = NOT_PROVIDED,
) -> Any:
    if isinstance(key, NotProvided) or not isinstance(value, NotProvided):
        return value

    return resolve(trace, key)
