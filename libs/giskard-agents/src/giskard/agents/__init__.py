from importlib import import_module
from typing import TYPE_CHECKING, Any

from giskard.core.utils import get_lib_version

if TYPE_CHECKING:
    from .chat import Chat
    from .context import RunContext
    from .embeddings import BaseEmbeddingModel, EmbeddingModel
    from .errors import Error, ModelRefusalError, WorkflowError
    from .generators import BaseGenerator, GenerationParams, Generator
    from .som import BaseSOM, SOMResponse, resolve_som
    from .templates import (
        MessageTemplate,
        add_prompts_path,
        get_prompts_manager,
        remove_prompts_path,
        set_default_prompts_path,
        set_template_policy,
        strict_templates,
    )
    from .tools import Tool, tool
    from .workflow import ChatWorkflow, ErrorPolicy, StepType, TemplateReference

_LAZY: dict[str, str] = {
    "Chat": ".chat",
    "RunContext": ".context",
    "BaseEmbeddingModel": ".embeddings",
    "EmbeddingModel": ".embeddings",
    "Error": ".errors",
    "ModelRefusalError": ".errors",
    "WorkflowError": ".errors",
    "BaseGenerator": ".generators",
    "GenerationParams": ".generators",
    "Generator": ".generators",
    "BaseSOM": ".som",
    "SOMResponse": ".som",
    "resolve_som": ".som",
    "MessageTemplate": ".templates",
    "add_prompts_path": ".templates",
    "get_prompts_manager": ".templates",
    "remove_prompts_path": ".templates",
    "set_default_prompts_path": ".templates",
    "set_template_policy": ".templates",
    "strict_templates": ".templates",
    "Tool": ".tools",
    "tool": ".tools",
    "ChatWorkflow": ".workflow",
    "ErrorPolicy": ".workflow",
    "StepType": ".workflow",
    "TemplateReference": ".workflow",
}

# Submodules reachable as attributes: every export's module, plus ``pipeline``.
_SUBMODULES = frozenset({module[1:] for module in _LAZY.values()} | {"pipeline"})

__version__ = get_lib_version("giskard-agents")

__all__ = [
    "__version__",
    "Generator",
    "BaseGenerator",
    "GenerationParams",
    "BaseSOM",
    "SOMResponse",
    "resolve_som",
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
    "set_template_policy",
    "strict_templates",
    "RunContext",
    "ErrorPolicy",
    "WorkflowError",
    "ModelRefusalError",
    "Error",
    "BaseEmbeddingModel",
    "EmbeddingModel",
    "StepType",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
