import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jinja2 import BaseLoader, Environment, StrictUndefined, nodes
from jinja2.exceptions import TemplateNotFound
from jinja2.ext import Extension
from jinja2.loaders import FileSystemLoader, PrefixLoader
from pydantic import BaseModel

from ..chat import Message


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
        return json.dumps(value.model_dump(), indent=4)
    return value


_inline_env = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
    autoescape=False,
    finalize=_finalize_value,
)


class MessageExtension(Extension):
    """Custom Jinja2 extension for parsing {% message role %}...{% endmessage %} blocks."""

    tags = {"message"}

    def __init__(self, environment):
        super().__init__(environment)
        if not hasattr(environment, "_collected_messages"):
            environment._collected_messages = []

    def parse(self, parser):
        """Parse a {% message role %}...{% endmessage %} block."""
        lineno = next(parser.stream).lineno
        role_node = parser.parse_expression()
        if isinstance(role_node, nodes.Name):
            role_node = nodes.Const(role_node.name)
        body = parser.parse_statements(["name:endmessage"], drop_needle=True)
        call_node = self.call_method("_handle_message", [role_node])

        return nodes.CallBlock(call_node, [], [], body).set_lineno(lineno)

    async def _handle_message(self, role: str, caller):
        """Handle a message block by rendering its content and storing it."""
        content = (await caller()).strip()
        self.environment._collected_messages.append(Message(role=role, content=content))
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


def create_message_environment(loader_mapping: dict[str, Path]) -> Environment:
    """Create a Jinja2 environment with MessageExtension."""
    return Environment(
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
