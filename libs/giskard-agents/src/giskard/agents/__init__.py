from giskard.core.utils import get_lib_version

from .chat import Chat
from .context import RunContext
from .embeddings import BaseEmbeddingModel, EmbeddingModel
from .errors import Error, ModelRefusalError, WorkflowError
from .generators import BaseGenerator, Generator
from .templates import (
    MessageTemplate,
    add_prompts_path,
    get_prompts_manager,
    remove_prompts_path,
    set_default_prompts_path,
    set_prompts_path,
)
from .tools import Tool, tool
from .workflow import ChatWorkflow, ErrorPolicy, StepType, TemplateReference

__version__ = get_lib_version("giskard-agents")

__all__ = [
    "__version__",
    "Generator",
    "BaseGenerator",
    "ChatWorkflow",
    "TemplateReference",
    "Chat",
    "Tool",
    "tool",
    "MessageTemplate",
    "set_prompts_path",
    "set_default_prompts_path",
    "add_prompts_path",
    "remove_prompts_path",
    "get_prompts_manager",
    "RunContext",
    "ErrorPolicy",
    "WorkflowError",
    "ModelRefusalError",
    "Error",
    "BaseEmbeddingModel",
    "EmbeddingModel",
    "StepType",
]
