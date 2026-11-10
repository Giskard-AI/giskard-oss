"""Template security policy.

Templates are rendered in a Jinja2 sandbox, which blocks private and internal
Python attributes but still lets a template call any public method of the
objects it receives — including pydantic APIs such as ``parse_file`` (reads
files from disk, optionally unpickling them) or ``model_validate_json``.

That is fine when templates are written by the person running them, which is
the default (``"permissive"``) policy. Services that render templates authored
by untrusted users should enable the ``"strict"`` policy, under which templates
can only *read data*:

- pydantic models expose their fields (including extra fields), computed fields
  and properties defined on the model's own classes; methods and the inherited
  ``BaseModel`` API are blocked;
- methods of other objects are blocked, except the built-in methods of plain
  data types (``str``, ``list``, ``dict``, ...) and Jinja's loop helpers;
  methods that subclasses of those types add or override (e.g. numpy scalars'
  ``tofile``) are blocked too;
- classes, classmethods and modules are never reachable, and their
  attributes cannot be read;
- items looked up with ``x[key]`` (or ``x.key`` falling back to an item) are
  checked like attributes, so containers cannot hand out modules, classes or
  callables;
- calls are checked as well, since a callable can be reached without any
  lookup (by iterating a container, through filters such as ``first``, or as
  a template variable): only the methods allowed above and Jinja's own
  callables (macros, blocks, ``loop``, ``range``, ``cycler``, ...) can be
  called;
- known mutable builtins (lists, dicts, sets) cannot be modified.

The policy is enabled process-wide with :func:`set_template_policy` (or the
``GISKARD_TEMPLATE_POLICY`` environment variable), or for a scope with
:func:`strict_templates`.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from types import BuiltinMethodType, MethodType, ModuleType
from typing import Any, Literal, cast, get_args
from weakref import WeakKeyDictionary

from jinja2.ext import Extension
from jinja2.runtime import BlockReference, LoopContext, Macro
from jinja2.runtime import TemplateReference as JinjaTemplateReference
from jinja2.sandbox import SandboxedEnvironment, modifies_known_mutable, safe_range
from jinja2.utils import Cycler, Joiner, Namespace, generate_lorem_ipsum
from pydantic import BaseModel

TemplatePolicy = Literal["permissive", "strict"]

_POLICY_ENV_VAR = "GISKARD_TEMPLATE_POLICY"


def _validate_policy(policy: str) -> TemplatePolicy:
    if isinstance(policy, str):
        policy = policy.strip().lower()
    if policy not in get_args(TemplatePolicy):
        raise ValueError(
            f"Unknown template policy {policy!r}; expected one of {get_args(TemplatePolicy)}"
        )
    return cast(TemplatePolicy, policy)


# Process-wide default. A ContextVar alone would not do: a value set at startup
# is not inherited by threads started afterwards, which would silently fall
# back to the permissive policy.
_default_policy: TemplatePolicy = _validate_policy(
    os.environ.get(_POLICY_ENV_VAR, "permissive")
)
_policy_override: ContextVar[TemplatePolicy | None] = ContextVar(
    "giskard_template_policy", default=None
)


def get_template_policy() -> TemplatePolicy:
    """Return the template policy in effect for the current context."""
    return _policy_override.get() or _default_policy


def set_template_policy(policy: TemplatePolicy) -> None:
    """Set the process-wide template policy.

    Parameters
    ----------
    policy : {"permissive", "strict"}
        ``"strict"`` restricts templates to reading data; use it whenever
        templates may come from untrusted users. Case and surrounding
        whitespace are ignored, as for ``GISKARD_TEMPLATE_POLICY``.
    """
    global _default_policy
    _default_policy = _validate_policy(policy)


@contextmanager
def strict_templates() -> Iterator[None]:
    """Render templates with the strict policy within this scope.

    The override is stored in a :class:`~contextvars.ContextVar`, so it
    follows ``await`` and :func:`asyncio.to_thread` (which copies the current
    context), but not a plain :class:`threading.Thread` or
    ``loop.run_in_executor`` started inside the scope: renders there fall
    back to the process-wide policy. Use :func:`set_template_policy` when
    renders may happen in such threads.
    """
    token = _policy_override.set("strict")
    try:
        yield
    finally:
        _policy_override.reset(token)


# Built-in data types whose built-in methods (``str.split``, ``dict.items``, ...)
# templates may call; mutating ones are rejected by ``modifies_known_mutable``.
_DATA_TYPES = (str, bytes, int, float, list, tuple, dict, set, frozenset)
# Jinja runtime helpers whose public methods (and, for ``self``, blocks) are
# part of the template language.
_JINJA_HELPERS = (LoopContext, Cycler, JinjaTemplateReference)
# Callables a template defines or gets from Jinja: macros (and ``caller``),
# blocks (``self.b()``, ``super()``), recursive ``loop()`` and joiners.
_JINJA_CALLABLES = (Macro, BlockReference, LoopContext, Joiner)
# The sandbox's default globals (with its bounded ``range``).
_JINJA_GLOBALS = (safe_range, dict, generate_lorem_ipsum, Cycler, Joiner, Namespace)
# The sandbox wraps ``str.format`` at lookup so that format fields are checked
# too; every such wrapper shares this code object.
_STR_FORMAT_WRAPPER_CODE = getattr(
    SandboxedEnvironment().wrap_str_format("".format), "__code__"
)
# Values that are never handed to a strict template, nor read from: classes
# expose constructors and class-level state, and modules expose everything
# they import (``os.environ``, ``sys.modules``, ...).
_OPAQUE_TYPES = (type, ModuleType)


# Weak keys so dynamically created models (e.g. tool argument models) can be
# garbage-collected.
_model_data_cache: WeakKeyDictionary[type[BaseModel], frozenset[str]] = (
    WeakKeyDictionary()
)


def _model_data_attrs(model_cls: type[BaseModel]) -> frozenset[str]:
    """Names of the fields, computed fields and own properties of a model."""
    attrs = _model_data_cache.get(model_cls)
    if attrs is None:
        names: set[str] = set()
        # Walk base-first so a subclass definition overrides its bases'; skip
        # pydantic's own classes.
        for klass in reversed(model_cls.__mro__):
            if klass in BaseModel.__mro__:
                continue
            for name, member in vars(klass).items():
                if isinstance(member, (property, cached_property)):
                    names.add(name)
                else:
                    names.discard(name)
        names.update(model_cls.model_fields, model_cls.model_computed_fields)
        attrs = _model_data_cache[model_cls] = frozenset(names)
    return attrs


def _is_opaque(value: Any) -> bool:
    """Whether ``value`` is a class, a classmethod or a module."""
    return isinstance(value, _OPAQUE_TYPES) or isinstance(
        getattr(value, "__self__", None), type
    )


def _is_builtin_data_method(obj: Any, attr: str, value: Any) -> bool:
    """Whether ``value`` is ``obj.attr`` resolving to a data type's own method.

    Subclasses of the data types are accepted (e.g. ``StrEnum`` members), but
    only for methods they inherit unchanged: numpy scalars subclass ``float``
    and ``str`` yet add methods such as ``tofile`` and ``dump`` that write
    files.
    """
    if not isinstance(value, BuiltinMethodType) or value.__self__ is not obj:
        return False
    base = next((t for t in _DATA_TYPES if isinstance(obj, t)), None)
    if base is None:
        return False
    builtin = vars(base).get(attr)
    resolved = next(
        (vars(klass)[attr] for klass in type(obj).__mro__ if attr in vars(klass)),
        None,
    )
    return builtin is not None and resolved is builtin


def is_strictly_safe_access(obj: Any, name: Any, value: Any) -> bool:
    """Decide whether a template may read ``obj.name`` or ``obj[name]``.

    Item lookups are checked like attribute reads, so containers cannot hand
    out what an attribute read would refuse.

    Parameters
    ----------
    obj : Any
        The object the attribute or item is read from.
    name : Any
        The attribute name or item key.
    value : Any
        The attribute or item value, already resolved.

    Returns
    -------
    bool
        ``True`` when the access is a plain data read under the strict policy.
    """
    if isinstance(obj, _OPAQUE_TYPES) or _is_opaque(value):
        return False
    if isinstance(obj, BaseModel):
        return not callable(value) and (
            name in _model_data_attrs(type(obj)) or name in (obj.model_extra or {})
        )
    if not callable(value) or isinstance(obj, _JINJA_HELPERS):
        return True
    return not modifies_known_mutable(obj, name) and _is_builtin_data_method(
        obj, name, value
    )


def is_strictly_safe_callable(value: Any) -> bool:
    """Decide whether a template may call ``value``.

    Lookups already refuse most callables, but iterating a container, filters
    such as ``first`` and template variables hand out values without one, so
    calls are checked too.

    Parameters
    ----------
    value : Any
        The object being called.

    Returns
    -------
    bool
        ``True`` when ``value`` is one of Jinja's own callables, or a method
        that :func:`is_strictly_safe_access` allows on its object.
    """
    if isinstance(value, _JINJA_CALLABLES) or any(value is f for f in _JINJA_GLOBALS):
        return True
    if getattr(value, "__code__", None) is _STR_FORMAT_WRAPPER_CODE:
        return True
    if not isinstance(value, (MethodType, BuiltinMethodType)):
        return False
    # Compiled templates call their extensions' methods (``{% message %}``).
    if isinstance(value.__self__, Extension):
        return True
    return not value.__name__.startswith("_") and is_strictly_safe_access(
        value.__self__, value.__name__, value
    )
