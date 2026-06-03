from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..workflow import WorkflowStep


class WorkflowError(RuntimeError):
    """An error that occurs during a workflow."""

    def __init__(
        self,
        message: str,
        *,
        exception: Exception | None = None,
        last_step: Optional["WorkflowStep"] = None,
    ):
        super().__init__(message)
        self.exception = exception
        self.last_step = last_step


class ModelRefusalError(RuntimeError):
    """Raised when the model explicitly refuses to generate content.

    Detected via ``finish_reason='refusal'`` or a non-null ``message.refusal``
    in the completion response. Propagates unwrapped through WorkflowError so
    callers can distinguish a deliberate refusal from other workflow failures.
    """

    def __init__(self, refusal: str | None = None) -> None:
        self.refusal = refusal
        super().__init__(refusal or "Model refused to generate content")
