from ._base import _BaseModel
from .chat import AssistantMessage
from .usage import Usage

# -- Chat Completion types -----------------------------------------------------


class Choice(_BaseModel):
    message: AssistantMessage
    finish_reason: str | None = None
    index: int = 0


class CompletionResponse(_BaseModel):
    choices: list[Choice]
    model: str | None = None
    usage: Usage | None = None
