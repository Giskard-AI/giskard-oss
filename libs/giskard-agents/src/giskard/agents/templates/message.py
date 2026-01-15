from typing import Any

from pydantic import BaseModel

from ..chat import Message, Role
from .environment import _inline_env


class MessageTemplate(BaseModel):
    role: Role
    content_template: str

    def render(self, **kwargs: Any) -> Message:
        """
        Render the message template with the given context.
        """
        template = _inline_env.from_string(self.content_template)
        rendered_content = template.render(**kwargs)

        return Message(role=self.role, content=rendered_content)
