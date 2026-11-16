from collections.abc import Collection
from typing import Any, NoReturn

from jinja2 import meta, nodes, pass_context
from jinja2.exceptions import TemplateAssertionError, TemplateSyntaxError

from .environment import _inline_env


@pass_context
def _not_folded(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError("templates are never rendered with _validation_env")


# Jinja's code generator evaluates constant expressions at compile time, even
# with ``optimized=False``, so compiling an untrusted template could run
# ``'a' * 10**10`` or ``'x' | center(10**9)``. Validation therefore compiles
# with an overlay of ``_inline_env`` (same syntax, globals and policy) that
# folds nothing costly: a sandboxed environment never folds the operators it
# intercepts, and Jinja never calls ``@pass_context`` filters or tests at
# compile time. The filters and tests keep their names, so compiling accepts
# and rejects exactly what ``_inline_env`` does. What can still be folded
# (literals, ``~``, comparisons, ``and``/``or``, subscripts) is linear in the
# length of the source.
_validation_env = _inline_env.overlay()
_validation_env.intercepted_binops = frozenset(_inline_env.binop_table)
_validation_env.intercepted_unops = frozenset(_inline_env.unop_table)
_validation_env.filters = _inline_env.filters.copy()
_validation_env.tests = _inline_env.tests.copy()
_validation_env.filters.update(dict.fromkeys(_inline_env.filters, _not_folded))
_validation_env.tests.update(dict.fromkeys(_inline_env.tests, _not_folded))

_LOADER_TAGS: tuple[tuple[type[nodes.Node], str], ...] = (
    (nodes.Include, "include"),
    (nodes.Import, "import"),
    (nodes.FromImport, "from"),
    (nodes.Extends, "extends"),
)

# Filters that take another filter or test by name, and the position of that
# name among their positional arguments.
_NAMED_CALLEES: dict[str, tuple[str, int]] = {
    "map": ("filter", 0),
    "select": ("test", 0),
    "reject": ("test", 0),
    "selectattr": ("test", 1),
    "rejectattr": ("test", 1),
}


class TemplateValidationError(ValueError):
    """Raised when a template could not be rendered as an inline message template."""

    def __init__(self, message: str, lineno: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.lineno = lineno


def validate_template(
    source: str,
    *,
    variables: Collection[str],
    required: Collection[str] = (),
) -> None:
    """Check that ``source`` is an inline template that can be rendered.

    Validation uses the same syntax, filters and tests as
    :class:`~giskard.agents.templates.MessageTemplate`, so it accepts exactly
    what rendering supports, regardless of the active template policy.
    Nothing is rendered or evaluated, not even constant expressions.

    Parameters
    ----------
    source : str
        The template source.
    variables : Collection[str]
        The only free variables the template may read.
    required : Collection[str]
        Variables the template must read.

    Raises
    ------
    ValueError
        If a name in ``required`` is not in ``variables`` (a caller bug).
    TemplateValidationError
        If the template is not valid, uses a loader tag (``include``,
        ``import``, ``from``, ``extends``), reads a variable outside
        ``variables`` or does not read a variable in ``required``.
    """
    unreachable = sorted(set(required) - set(variables))
    if unreachable:
        raise ValueError(
            f"required variables not in variables: {', '.join(unreachable)}"
        )
    try:
        _validate(source, variables, required)
    except (SyntaxError, RecursionError) as error:
        # Jinja compiles to Python source: deep nesting overflows the parser
        # or the interpreter stack before Jinja can report anything.
        raise TemplateValidationError(
            "not a valid template: nested too deeply"
        ) from error
    except (MemoryError, OverflowError) as error:
        # Nothing is evaluated, but a memory-capped process can still run out.
        raise TemplateValidationError(
            "not a valid template: too large to compile"
        ) from error


def _validate(
    source: str, variables: Collection[str], required: Collection[str]
) -> None:
    try:
        ast = _validation_env.parse(source)
    except TemplateSyntaxError as error:
        raise TemplateValidationError(
            f"not a valid template: {error.message}", error.lineno
        ) from error
    except ValueError as error:
        # The lexer's int() rejects literals over Python's digit limit.
        raise TemplateValidationError(f"not a valid template: {error}") from error

    for node_type, tag in _LOADER_TAGS:
        found = ast.find(node_type)
        if found is not None:
            raise TemplateValidationError(
                f"'{tag}' tags are not allowed: the template has no loader",
                found.lineno,
            )

    try:
        _validation_env.compile(ast)
    except TemplateAssertionError as error:
        raise TemplateValidationError(
            f"not a valid template: {error.message}", error.lineno
        ) from error

    # Jinja defers unknown filters and tests inside ``if`` blocks and inline
    # conditionals to render time, so compiling alone does not catch them.
    for node in ast.find_all((nodes.Filter, nodes.Test)):
        for kind, name in _callee_names(node):
            known = (
                _validation_env.filters if kind == "filter" else _validation_env.tests
            )
            if name not in known:
                raise TemplateValidationError(
                    f"not a valid template: No {kind} named {name!r}.", node.lineno
                )

    # Excludes environment globals (``range``, ``namespace``...). Compiles
    # ``ast`` again, with the environment that parsed it.
    used = meta.find_undeclared_variables(ast)
    extra = used - set(variables)
    if extra:
        raise TemplateValidationError(
            f"may only use the variables {', '.join(sorted(variables))}, "
            f"found: {', '.join(sorted(extra))}"
        )
    for name in required:
        if name not in used:
            raise TemplateValidationError(f"must use the variable '{name}'")


def _callee_names(node: nodes.Filter | nodes.Test) -> list[tuple[str, str]]:
    """Kind and name of ``node``, plus any filter or test it takes by name."""
    names = [("filter" if isinstance(node, nodes.Filter) else "test", node.name)]
    callee = _NAMED_CALLEES.get(node.name) if isinstance(node, nodes.Filter) else None
    if callee is None:
        return names
    kind, position = callee
    if len(node.args) > position:
        arg = node.args[position]
        if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
            names.append((kind, arg.value))
    return names
