from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Template
from pydantic import BaseModel, Field
from typing_extensions import deprecated

from ..chat import Message
from .environment import create_message_environment


async def render_messages_template(
    template: Template, variables: Dict[str, Any] = None
) -> List[Message]:
    """
    Render a template and collect any messages defined with {% message %} blocks.

    Parameters
    ----------
    template : Template
        The Jinja2 template to render
    variables : Dict[str, Any], optional
        Variables to pass to the template

    Returns
    -------
    List[Message]
        List of parsed Message objects
    """
    rendered_output = await template.render_async(variables or {})
    messages = template.environment._collected_messages

    # Two cases here:
    # 1. There are message blocks. In this case, the render output must be empty (at most whitespaces).
    # 2. There are no message blocks. In this case, we will create a single user message with the rendered output.
    if messages:
        if rendered_output.strip():
            raise ValueError(
                "Template contains message blocks but rendered output is not empty."
            )
        return messages
    else:
        return [Message(role="user", content=rendered_output)]


class PromptsManager(BaseModel):
    """Manages prompts path and template loading."""

    default_prompts_path: Path = Field(default_factory=lambda: Path.cwd() / "prompts")

    namespaces: Dict[str, Path] = Field(default_factory=dict)

    def set_default_prompts_path(self, path: str | Path):
        """Set a custom prompts path."""
        self.default_prompts_path = Path(path)

    def add_prompts_path(self, path: str | Path, namespace: str):
        """Add a custom prompts path for a given namespace."""
        if namespace in self.namespaces:
            raise ValueError(f"Namespace {namespace} already exists")
        self.namespaces[namespace] = Path(path)

    def remove_prompts_path(self, namespace: str):
        """Remove a custom prompts path for a given namespace."""
        if namespace not in self.namespaces:
            raise ValueError(f"Namespace {namespace} does not exist")
        del self.namespaces[namespace]

    @deprecated("Use set_default_prompts_path instead")
    def set_prompts_path(self, path: str | Path):
        self.set_default_prompts_path(path)

    async def render_template(
        self, template_name: str, variables: Dict[str, Any] = None
    ) -> List[Message]:
        """
        Load and parse a template file, returning a list of Message objects.

        Parameters
        ----------
        template_name : str
            The template name
        variables : Dict[str, Any], optional
            Variables to pass to the template for rendering

        Returns
        -------
        List[Message]
            List of parsed Message objects
        """
        # We create a fresh environment for each render to isolate the state
        # between renders. This is slightly inefficient but necessary for the
        # message parser to work correctly.
        env = create_message_environment(
            {
                "__default__": self.default_prompts_path,
                **self.namespaces,
            }
        )
        template = env.get_template(template_name)

        messages = await render_messages_template(template, variables)

        return messages


# Global instance
_prompts_manager = PromptsManager()


def get_prompts_manager() -> PromptsManager:
    """Get the global prompts manager."""
    return _prompts_manager


@deprecated("Use set_default_prompts_path instead")
def set_prompts_path(path: str):
    """Set a custom prompts path."""
    _prompts_manager.set_default_prompts_path(path)


def set_default_prompts_path(path: str):
    """Set a custom prompts path."""
    _prompts_manager.set_default_prompts_path(path)


def add_prompts_path(path: str, namespace: str):
    """Add a custom prompts path for a given namespace."""
    _prompts_manager.add_prompts_path(path, namespace)


def remove_prompts_path(namespace: str):
    """Remove a custom prompts path for a given namespace."""
    _prompts_manager.remove_prompts_path(namespace)


async def render_template(
    template_name: str, variables: Dict[str, Any] = None
) -> list[Message]:
    """Load and parse a template file."""
    return await _prompts_manager.render_template(template_name, variables)
