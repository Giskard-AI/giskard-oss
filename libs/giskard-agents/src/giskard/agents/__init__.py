from giskard.core.utils import get_lib_version

from .chat import Chat
from .context import RunContext
from .embeddings import (
    BaseEmbeddingModel,
    EmbeddingModel,
    GiskardLLMEmbeddingModel,
    LiteLLMEmbeddingModel,
)
from .errors import Error, ModelRefusalError, WorkflowError
from .generators import BaseGenerator, Generator
from .resolve import resolve_embedding_model, resolve_generator
from .templates import (
    MessageTemplate,
    add_prompts_path,
    get_prompts_manager,
    remove_prompts_path,
    set_default_prompts_path,
)
from .tools import Tool, tool
from .workflow import ChatWorkflow, ErrorPolicy, StepType, TemplateReference

__version__ = get_lib_version("giskard-agents")

__all__ = [
    "__version__",
    "Generator",
    "BaseGenerator",
    "resolve_generator",
    "resolve_embedding_model",
    "ChatWorkflow",
    "TemplateReference",
    "Chat",
    "Tool",
    "tool",
    "MessageTemplate",
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
    "GiskardLLMEmbeddingModel",
    "LiteLLMEmbeddingModel",
    "StepType",
]
