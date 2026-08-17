from ._base import _BaseModel
from .chat import AssistantMessage
from .usage import Usage

# -- Chat Completion types -----------------------------------------------------


class Choice(_BaseModel):
    """One completion candidate returned by a chat-completion call."""

    message: AssistantMessage
    finish_reason: str | None = None
    index: int = 0


class CompletionResponse(_BaseModel):
    """Provider-agnostic result of a chat-completion call."""

    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None
