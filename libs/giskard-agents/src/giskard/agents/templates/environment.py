import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from giskard.llm import chat
from jinja2 import BaseLoader, StrictUndefined, nodes
from jinja2.exceptions import TemplateNotFound
from jinja2.ext import Extension
from jinja2.loaders import FileSystemLoader, PrefixLoader
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel


@runtime_checkable
class LLMFormattable(Protocol):
    """Protocol for objects that can format themselves for LLM consumption."""

    def _repr_prompt_(self) -> str:
        """Format the object for LLM consumption.

        Returns
        -------
        str
            The formatted string representation of the object.
        """
        ...


class Trusted(str):
    """A string that is rendered into prompts verbatim, without escaping.

    Prompt templates escape every interpolated value by default (see
    :func:`_finalize_value`). Values wrapped in ``Trusted`` opt out of that
    escaping, and must therefore never be derived from untrusted input.
    """


def _render_value(value: Any) -> str:
    if isinstance(value, LLMFormattable):
        value = value._repr_prompt_()
    elif isinstance(value, BaseModel):
        return json.dumps(value.model_dump(mode="json"), indent=4)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _finalize_value(value: Any) -> str:
    """Render an interpolated template expression, escaping it by default.

    Prompts embed untrusted data (agent outputs, traces, documents) inside
    pseudo-XML markers such as ``<AGENT ANSWER>...</AGENT ANSWER>``. The
    environment renders prompts with ``autoescape=False``, so unescaped data
    could contain a literal closing marker and inject instructions into the
    judge. Escaping ``&``, ``<`` and ``>`` for every value makes the markers
    unforgeable while keeping the text human- and model-readable; a template
    that genuinely needs raw output opts out with :class:`Trusted` (available
    in templates as the ``trusted`` filter).

    Parameters
    ----------
    value : Any
        The interpolated value. ``None`` renders as an empty string.

    Returns
    -------
    str
        The rendered text, escaped unless the value is :class:`Trusted`.
    """
    if isinstance(value, Trusted):
        return str(value)
    return _escape(_render_value(value))


def trusted(value: Any) -> Trusted:
    """Mark a value as safe to render verbatim, bypassing prompt escaping.

    Only use on values that cannot be influenced by untrusted input, such as
    generated output-format instructions.

    Parameters
    ----------
    value : Any
        The trusted value. Rendered like a normal template expression;
        ``None`` renders as an empty string.

    Returns
    -------
    Trusted
        The rendered text, exempt from escaping.
    """
    return value if isinstance(value, Trusted) else Trusted(_render_value(value))


def fence(value: Any) -> Trusted:
    """Escape a value explicitly.

    Kept for backwards compatibility with templates written before escaping
    became the default; :func:`_finalize_value` now escapes every interpolated
    value, so this filter is a no-op in practice.

    Parameters
    ----------
    value : Any
        The (untrusted) value to embed in a prompt. ``None`` renders as an
        empty string.

    Returns
    -------
    Trusted
        The escaped text, marked so it is not escaped a second time.
    """
    return Trusted(_escape(_render_value(value)))


_inline_env = SandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
    autoescape=False,
    finalize=_finalize_value,
)
_inline_env.filters["fence"] = fence
_inline_env.filters["trusted"] = trusted


class MessageExtension(Extension):
    """Custom Jinja2 extension for parsing {% message role %}...{% endmessage %} blocks."""

    tags = {"message"}

    def __init__(self, environment):
        super().__init__(environment)
        if not hasattr(environment, "_collected_messages"):
            environment._collected_messages = []  # pyright: ignore[reportAttributeAccessIssue]

    def parse(self, parser):
        """Parse a {% message role %}...{% endmessage %} block."""
        lineno = next(parser.stream).lineno
        role_node = parser.parse_expression()
        if isinstance(role_node, nodes.Name):
            role_node = nodes.Const(role_node.name)
        body = parser.parse_statements(("name:endmessage",), drop_needle=True)
        call_node = self.call_method("_handle_message", [role_node])

        return nodes.CallBlock(call_node, [], [], body).set_lineno(lineno)

    async def _handle_message(
        self, role: Literal["user", "assistant", "system", "developer"], caller
    ):
        """Handle a message block by rendering its content and storing it."""
        content = (await caller()).strip()
        self.environment._collected_messages.append(chat.message(content, role))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return ""


class PromptsLoader(PrefixLoader):
    def get_loader(self, template: str) -> tuple[BaseLoader, str]:
        try:
            prefix, name = template.split(self.delimiter, 1)
        except ValueError:
            prefix = "__default__"
            name = template

        try:
            loader = self.mapping[prefix]
        except KeyError as e:
            raise TemplateNotFound(template) from e

        return loader, name


def create_message_environment(loader_mapping: dict[str, Path]) -> SandboxedEnvironment:
    """Create a Jinja2 environment with MessageExtension."""
    env = SandboxedEnvironment(
        loader=PromptsLoader(
            {
                namespace: FileSystemLoader(path)
                for namespace, path in loader_mapping.items()
            },
            delimiter="::",
        ),
        extensions=[MessageExtension],
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
        autoescape=False,
        enable_async=True,
        finalize=_finalize_value,
    )
    env.filters["fence"] = fence
    env.filters["trusted"] = trusted
    return env
