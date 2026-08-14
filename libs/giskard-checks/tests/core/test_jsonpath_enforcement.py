"""Enforcement test: all JSONPath fields in Check subclasses must use JSONPathStr."""

import re
import types
from collections.abc import Iterator
from typing import Annotated, Union, get_args, get_origin

# Import every subpackage that defines checks, so the ``Check.__subclasses__``
# walk below actually sees them. These imports must stay explicit: relying on
# ``giskard.checks.__init__`` to pull them in transitively means slimming that
# package would silently drop whole families of checks from this test while it
# kept reporting green.
import giskard.checks.builtin  # noqa: F401 - registers the builtin checks
import giskard.checks.judges  # noqa: F401 - registers the LLM judge checks
from giskard.checks.core.check import Check
from giskard.checks.core.extraction import _JSONPathStrMarker
from pydantic import BaseModel
from pydantic.fields import FieldInfo

JSONPATH_FIELD = re.compile(r"^(key|.+_key)$")

# Only classes defined inside the library are subject to this convention. A
# user's or third-party plugin's Check subclass may legitimately name a field
# ``key`` without using JSONPathStr, and must not fail giskard's own suite.
LIBRARY_MODULE_PREFIX = "giskard.checks"

# Every check family that must be covered by the walk. Guards against an import
# above being dropped, or a subpackage being added without being imported here.
EXPECTED_CHECK_MODULE_PREFIXES = (
    "giskard.checks.builtin",
    "giskard.checks.judges",
)


def _walk_subclasses(cls: type) -> Iterator[type]:
    """Recursively yield all subclasses of cls, concrete and abstract."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_subclasses(sub)


def _is_generic_parametrization(cls: type) -> bool:
    """Return True for pydantic generic aliases like ``Check[Any, Any, Trace]``.

    Pydantic creates a real class object for each parametrization of a generic
    model, and those show up in ``__subclasses__()``. They share the fields of
    the class they parametrize, so reporting them adds noise (the same violation
    named several times, under aliases like ``BaseLLMCheck[TypeVar, TypeVar,
    TypeVar]`` that appear nowhere in the source).
    """
    return bool(getattr(cls, "__pydantic_generic_metadata__", {}).get("args"))


def _library_check_classes() -> dict[tuple[str, str], type[BaseModel]]:
    """Return the deduplicated, in-library source classes to enforce against.

    Filters out generic parametrizations (which duplicate their origin class)
    and classes defined outside ``giskard.checks`` (user code and test-local
    subclasses), then dedupes by module + qualname.

    ``Check`` itself is included: ``__subclasses__()`` never yields the root, so
    a JSONPath field added to the base class would otherwise go unchecked.
    """
    classes: dict[tuple[str, str], type[BaseModel]] = {}
    for cls in (Check, *_walk_subclasses(Check)):
        if _is_generic_parametrization(cls):
            continue
        if not cls.__module__.startswith(LIBRARY_MODULE_PREFIX):
            continue
        if not issubclass(cls, BaseModel):
            continue
        classes.setdefault((cls.__module__, cls.__qualname__), cls)
    return classes


def _annotation_has_marker(annotation: object) -> bool:
    """Recursively check if an annotation contains _JSONPathStrMarker.

    This handles complex type annotations like:
    - Annotated[str, AfterValidator(...), _JSONPathStrMarker()]
    - JSONPathStr | None
    - JSONPathStr | MISSING
    """
    if get_origin(annotation) is Annotated:
        return any(isinstance(m, _JSONPathStrMarker) for m in get_args(annotation)[1:])
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        return any(_annotation_has_marker(arg) for arg in get_args(annotation))
    return False


def _has_jsonpath_marker(field_info: FieldInfo) -> bool:
    """Return True if the field uses JSONPathStr.

    Pydantic v2 stores Annotated metadata in two places depending on the type:
    - Simple `JSONPathStr`: marker is in field_info.metadata
    - Union `JSONPathStr | None` / `JSONPathStr | MISSING`: marker is
      inside field_info.annotation (the Annotated[str, ...] is preserved within
      the Union at the annotation level)
    """
    if any(isinstance(m, _JSONPathStrMarker) for m in field_info.metadata):
        return True
    return _annotation_has_marker(field_info.annotation)


def _violations_in(classes: dict[tuple[str, str], type[BaseModel]]) -> list[str]:
    """Return a formatted violation line for each offending field."""
    violations: list[str] = []
    for (module, qualname), cls in sorted(classes.items()):
        for field_name, field_info in cls.model_fields.items():
            if JSONPATH_FIELD.match(field_name) and not _has_jsonpath_marker(
                field_info
            ):
                violations.append(
                    f"{module}.{qualname}.{field_name}: "
                    f"annotation={field_info.annotation!r}"
                )
    return violations


def test_check_walk_covers_every_check_subpackage():
    """Guard the coverage of the enforcement walk itself.

    ``test_all_jsonpath_fields_use_jsonpath_str`` can only catch violations in
    classes it actually sees. If an import at the top of this module is dropped,
    or a new check subpackage is added without being imported here, the
    enforcement test would keep passing while silently covering less. This test
    fails instead.
    """
    modules = {module for module, _ in _library_check_classes()}
    for prefix in EXPECTED_CHECK_MODULE_PREFIXES:
        assert any(module.startswith(prefix) for module in modules), (
            f"No Check subclass from {prefix!r} is visible to the enforcement "
            f"walk. Add an explicit `import {prefix}` at the top of this module "
            f"so its checks are covered.\nVisible modules: {sorted(modules)}"
        )


def test_enforcement_walk_is_scoped_and_deduplicated():
    """The walk reports real source classes, not aliases or foreign classes."""
    classes = _library_check_classes()

    # No pydantic generic parametrizations: names must be findable in the source.
    assert not [qualname for _, qualname in classes if "[" in qualname], (
        f"Generic parametrizations leaked into the walk: {sorted(classes)}"
    )

    # Nothing from outside the library (user code, plugins, test-local classes).
    assert all(module.startswith(LIBRARY_MODULE_PREFIX) for module, _ in classes), (
        f"Out-of-library classes leaked into the walk: {sorted(classes)}"
    )


def test_all_jsonpath_fields_use_jsonpath_str():
    """Enforce that all JSONPath fields use the JSONPathStr type.

    This architectural constraint ensures:
    1. JSONPath syntax is validated at model creation time
    2. All JSONPath expressions start with 'trace.' prefix
    3. API consistency across all Check subclasses
    4. Better error messages for users when they provide invalid paths
    """
    violations = _violations_in(_library_check_classes())
    assert not violations, (
        "The following JSONPath fields do not use JSONPathStr.\n"
        "All fields named 'key' or ending in '_key' must be annotated as JSONPathStr "
        "(or JSONPathStr | None / JSONPathStr | MISSING):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
