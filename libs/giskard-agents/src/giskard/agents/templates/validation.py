from collections.abc import Collection

from jinja2 import meta, nodes
from jinja2.exceptions import TemplateAssertionError, TemplateSyntaxError

from .environment import _inline_env

_LOADER_TAGS: tuple[tuple[type[nodes.Node], str], ...] = (
    (nodes.Include, "include"),
    (nodes.Import, "import"),
    (nodes.FromImport, "from"),
    (nodes.Extends, "extends"),
)


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

    Validation uses the same environment as
    :class:`~giskard.agents.templates.MessageTemplate`, so it accepts exactly
    the syntax, filters and tests that rendering supports, regardless of the
    active template policy. Nothing is rendered.

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


def _validate(
    source: str, variables: Collection[str], required: Collection[str]
) -> None:
    try:
        ast = _inline_env.parse(source)
    except TemplateSyntaxError as error:
        raise TemplateValidationError(
            f"not a valid template: {error.message}", error.lineno
        ) from error

    for node_type, tag in _LOADER_TAGS:
        found = ast.find(node_type)
        if found is not None:
            raise TemplateValidationError(
                f"'{tag}' tags are not allowed: the template has no loader",
                found.lineno,
            )

    try:
        _inline_env.compile(ast)
    except TemplateAssertionError as error:
        raise TemplateValidationError(
            f"not a valid template: {error.message}", error.lineno
        ) from error

    # Jinja defers unknown filters and tests inside ``if`` blocks and inline
    # conditionals to render time, so compiling alone does not catch them.
    for node in ast.find_all((nodes.Filter, nodes.Test)):
        is_filter = isinstance(node, nodes.Filter)
        known = _inline_env.filters if is_filter else _inline_env.tests
        if node.name not in known:
            kind = "filter" if is_filter else "test"
            raise TemplateValidationError(
                f"not a valid template: No {kind} named {node.name!r}.", node.lineno
            )

    used = meta.find_undeclared_variables(ast) - set(_inline_env.globals)
    extra = used - set(variables)
    if extra:
        raise TemplateValidationError(
            f"may only use the variables {', '.join(sorted(variables))}, "
            f"found: {', '.join(sorted(extra))}"
        )
    for name in required:
        if name not in used:
            raise TemplateValidationError(f"must use the variable '{name}'")
