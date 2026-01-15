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
