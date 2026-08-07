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


def _finalize_value(value: Any) -> Any:
    if isinstance(value, LLMFormattable):
        return value._repr_prompt_()
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump(mode="json"), indent=4)
    return value


def fence(value: Any) -> str:
    """Neutralize delimiter breakouts in untrusted prompt content.

    Prompts embed untrusted data (agent outputs, traces) inside pseudo-XML
    markers such as ``<AGENT ANSWER>...</AGENT ANSWER>``. Because the
    environment renders prompts with ``autoescape=False``, unescaped data could
    contain a literal closing marker and inject instructions into the judge.
    Rendering the value the same way the engine would (via ``_finalize_value``)
    and escaping ``&``, ``<`` and ``>`` makes the markers unforgeable while
    keeping the text human- and model-readable.

    Parameters
    ----------
    value : Any
        The (untrusted) value to embed in a prompt. Finalized like a normal
        template expression before escaping; ``None`` renders as an empty
        string.

    Returns
    -------
    str
        The finalized text with ``&``, ``<`` and ``>`` replaced by their HTML
        entities.
    """
    finalized = _finalize_value(value)
    if finalized is None:
        return ""
    text = finalized if isinstance(finalized, str) else str(finalized)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_inline_env = SandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
    autoescape=False,
    finalize=_finalize_value,
)
_inline_env.filters["fence"] = fence


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
    return env
