"""Enforcement test: all JSONPath fields in Check subclasses must use JsonPathStr."""

import re
import types
from typing import Annotated, Union, get_args, get_origin

import giskard.checks.builtin  # noqa: F401 - triggers all @Check.register imports
from giskard.checks.core.check import Check
from giskard.checks.core.extraction import _JsonPathStrMarker

JSONPATH_FIELD = re.compile(r"^(key|.+_key)$")


def _all_check_subclasses(cls):
    """Recursively yield all concrete subclasses of cls."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_check_subclasses(sub)


def _annotation_has_marker(annotation) -> bool:
    """Recursively check if an annotation contains _JsonPathStrMarker."""
    if get_origin(annotation) is Annotated:
        return any(isinstance(m, _JsonPathStrMarker) for m in get_args(annotation)[1:])
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        return any(_annotation_has_marker(arg) for arg in get_args(annotation))
    return False


def _has_jsonpath_marker(field_info) -> bool:
    """Return True if the field uses JsonPathStr.

    Pydantic v2 stores Annotated metadata in two places depending on the type:
    - Simple `JsonPathStr`: marker is in field_info.metadata
    - Union `JsonPathStr | None` / `JsonPathStr | NotProvided`: marker is
      inside field_info.annotation (the Annotated[str, ...] is preserved within
      the Union at the annotation level)
    """
    if any(isinstance(m, _JsonPathStrMarker) for m in field_info.metadata):
        return True
    return _annotation_has_marker(field_info.annotation)


def test_all_jsonpath_fields_use_jsonpath_str():
    violations = []
    for cls in _all_check_subclasses(Check):
        if not hasattr(cls, "model_fields"):
            continue
        for field_name, field_info in cls.model_fields.items():
            if JSONPATH_FIELD.match(field_name):
                if not _has_jsonpath_marker(field_info):
                    violations.append(
                        f"{cls.__name__}.{field_name}: {field_info.annotation}"
                    )
    assert not violations, (
        "The following JSONPath fields do not use JsonPathStr.\n"
        "All fields named 'key' or ending in '_key' must be annotated as JsonPathStr "
        "(or JsonPathStr | None / JsonPathStr | NotProvided):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
