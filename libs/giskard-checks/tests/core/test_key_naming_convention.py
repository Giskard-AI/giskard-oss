"""Enforcement test: JSONPath field names follow the ``{static}_key`` convention.

``test_jsonpath_enforcement`` guards the *type* of JSONPath fields. This module
guards their *names*, which is the half a reader relies on to guess a check's
API without opening it:

a) the field naming the value under test is always ``target_key``;
b) every other JSONPath field is ``{static}_key``, where ``{static}`` is the
   name of a sibling static field on the same model holding the literal value
   that key would otherwise be resolved from.

Rule (b) is what makes ``reference_text_key`` / ``reference_text``,
``keyword_key`` / ``keyword`` and ``context_key`` / ``context`` predictable
rather than a set of independently invented spellings. The static half is named
for meaning; the key half is only ever that name plus ``_key``, so neither can
drift from the other under later edits.

The walk deliberately reuses the helpers in ``test_jsonpath_enforcement``
(generic-parametrization dedup, library-module scoping, ``Check`` itself
included) so both enforcement tests cover exactly the same set of classes; a
class hidden from one would otherwise be silently exempt from the other.
"""

import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _load_enforcement_helpers() -> tuple[
    Callable[[FieldInfo], bool],
    Callable[[], Mapping[tuple[str, str], type[BaseModel]]],
]:
    """Import the sibling enforcement module by path.

    The test tree has no ``__init__.py``, so ``test_jsonpath_enforcement`` is
    neither a relative import (no parent package) nor a name pyright can resolve
    from a bare ``import``. Loading it by path reuses the real helpers instead of
    duplicating the walk, which is the point: both enforcement tests must cover
    exactly the same set of classes, or a class hidden from one is silently
    exempt from the other.
    """
    path = Path(__file__).with_name("test_jsonpath_enforcement.py")
    spec = importlib.util.spec_from_file_location("_jsonpath_enforcement", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        cast(Callable[[FieldInfo], bool], module._has_jsonpath_marker),
        cast(
            Callable[[], Mapping[tuple[str, str], type[BaseModel]]],
            module._library_check_classes,
        ),
    )


_has_jsonpath_marker, _library_check_classes = _load_enforcement_helpers()

# The canonical name for the subject of a check: the value under test. It is
# the one JSONPath field exempt from rule (b), because the thing it points at
# is the trace itself, not a literal a user could have written inline.
SUBJECT_FIELD = "target_key"

KEY_SUFFIX = "_key"


def _jsonpath_fields(cls: type[BaseModel]) -> list[str]:
    return [
        name for name, info in cls.model_fields.items() if _has_jsonpath_marker(info)
    ]


def _all_jsonpath_fields() -> list[tuple[str, str, str, type[BaseModel]]]:
    """Flatten the walk into (module, qualname, field_name, cls) rows."""
    return [
        (module, qualname, field, cls)
        for (module, qualname), cls in sorted(_library_check_classes().items())
        for field in _jsonpath_fields(cls)
    ]


def test_walk_sees_jsonpath_fields_at_all() -> None:
    """Guard the guard: an empty walk would make every assertion below vacuous."""
    rows = _all_jsonpath_fields()

    assert len(rows) > 10, (
        "The convention walk found almost no JSONPath fields, so the naming "
        f"assertions below prove nothing. Rows: {rows}"
    )


def test_every_jsonpath_field_ends_in_key() -> None:
    """Rule (a): a JSONPath field is ``target_key`` or ``{something}_key``.

    ``{something}`` must be non-empty, so a field named exactly ``_key`` (or the
    bare ``key``) is a violation: neither tells a reader what the path
    points at.
    """
    violations = [
        f"{module}.{qualname}.{field}"
        for module, qualname, field, _ in _all_jsonpath_fields()
        if field != SUBJECT_FIELD
        and not (field.endswith(KEY_SUFFIX) and len(field) > len(KEY_SUFFIX))
    ]

    assert not violations, (
        "JSONPath fields must be named 'target_key' (the value under test) or "
        "'{static}_key' with a non-empty '{static}':\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_every_reference_key_has_a_static_sibling() -> None:
    """Rule (b): ``{static}_key`` requires a sibling static field ``{static}``.

    Every JSONPath field other than ``target_key`` names a value the user could
    also supply literally. The literal field must exist and must be exactly the
    key's name with ``_key`` stripped, so ``reference_text_key`` pairs with
    ``reference_text`` and ``context_key`` with ``context``.
    """
    violations: list[str] = []
    for module, qualname, field, cls in _all_jsonpath_fields():
        if field == SUBJECT_FIELD or not field.endswith(KEY_SUFFIX):
            continue
        static = field[: -len(KEY_SUFFIX)]
        if static not in cls.model_fields:
            violations.append(
                f"{module}.{qualname}.{field}: expected a sibling static field "
                f"{static!r}, but the model has {sorted(cls.model_fields)}"
            )

    assert not violations, (
        "Every JSONPath field besides 'target_key' must pair with a sibling "
        "static field of the same name minus '_key':\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_static_sibling_is_not_itself_a_jsonpath_field() -> None:
    """The sibling holds a literal value, not another path.

    A ``{static}`` annotated ``JSONPathStr`` would mean a ``{static}_key``
    pointing at a path — nonsense that rule (b) alone would still accept.
    """
    violations: list[str] = []
    for module, qualname, field, cls in _all_jsonpath_fields():
        if field == SUBJECT_FIELD or not field.endswith(KEY_SUFFIX):
            continue
        static = field[: -len(KEY_SUFFIX)]
        sibling = cls.model_fields.get(static)
        if sibling is not None and _has_jsonpath_marker(sibling):
            violations.append(f"{module}.{qualname}.{static}")

    assert not violations, (
        "These static siblings are themselves JSONPath fields:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
