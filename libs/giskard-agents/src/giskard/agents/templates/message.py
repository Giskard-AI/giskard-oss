from typing import Any

from pydantic import BaseModel

from ..chat import Message, Role
from .environment import _inline_env


class MessageTemplate(BaseModel):
    """Inline Jinja2 message body before a workflow run.

    ``content_template`` is compiled and rendered with Jinja2 (e.g. via
    :meth:`render` or when the workflow resolves messages).

    .. warning::

        Only use template strings from trusted sources. If ``content_template`` can
        be influenced by untrusted input, it can lead to template injection and
        unintended disclosure or execution of logic exposed by the template
        environment.
    """

    role: Role
    content_template: str

    def render(self, **kwargs: Any) -> Message:
        """
        Render the message template with the given context.

        The template is evaluated as Jinja2; do not pass untrusted values in
        ``content_template`` (see class docstring).
        """
        template = _inline_env.from_string(self.content_template)
        rendered_content = template.render(**kwargs)

        return Message(role=self.role, content=rendered_content)
