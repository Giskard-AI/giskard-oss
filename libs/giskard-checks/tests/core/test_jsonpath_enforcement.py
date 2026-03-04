"""Enforcement test: all JSONPath fields in Check subclasses must use JSONPathStr."""

import re
import types
from typing import Annotated, Union, get_args, get_origin

import giskard.checks.builtin  # noqa: F401 - triggers all @Check.register imports
from giskard.checks.core.check import Check
from giskard.checks.core.extraction import _JSONPathStrMarker

JSONPATH_FIELD = re.compile(r"^(key|.+_key)$")


def _all_check_subclasses(cls):
    """Recursively yield all concrete and abstract subclasses of cls.

    This ensures all Check subclasses follow the JSONPathStr convention,
    even if they're abstract base classes.
    """
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_check_subclasses(sub)


def _annotation_has_marker(annotation) -> bool:
    """Recursively check if an annotation contains _JSONPathStrMarker.

    This handles complex type annotations like:
    - Annotated[str, AfterValidator(...), _JSONPathStrMarker()]
    - JSONPathStr | None
    - JSONPathStr | NotProvided
    """
    if get_origin(annotation) is Annotated:
        return any(isinstance(m, _JSONPathStrMarker) for m in get_args(annotation)[1:])
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        return any(_annotation_has_marker(arg) for arg in get_args(annotation))
    return False


def _has_jsonpath_marker(field_info) -> bool:
    """Return True if the field uses JSONPathStr.

    Pydantic v2 stores Annotated metadata in two places depending on the type:
    - Simple `JSONPathStr`: marker is in field_info.metadata
    - Union `JSONPathStr | None` / `JSONPathStr | NotProvided`: marker is
      inside field_info.annotation (the Annotated[str, ...] is preserved within
      the Union at the annotation level)
    """
    if any(isinstance(m, _JSONPathStrMarker) for m in field_info.metadata):
        return True
    return _annotation_has_marker(field_info.annotation)


def test_all_jsonpath_fields_use_jsonpath_str():
    """Enforce that all JSONPath fields use the JSONPathStr type.

    This architectural constraint ensures:
    1. JSONPath syntax is validated at model creation time
    2. All JSONPath expressions start with 'trace.' prefix
    3. API consistency across all Check subclasses
    4. Better error messages for users when they provide invalid paths
    """
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
        "The following JSONPath fields do not use JSONPathStr.\n"
        "All fields named 'key' or ending in '_key' must be annotated as JSONPathStr "
        "(or JSONPathStr | None / JSONPathStr | NotProvided):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
