"""giskard-llm -- lightweight LLM routing over native provider SDKs."""

from . import chat
from .errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    ProviderNotAvailableError,
    RateLimitError,
    ServerError,
    UnsupportedOperationError,
)
from .providers.base import CompletionProvider, EmbeddingProvider, ResponseProvider
from .retry import should_retry
from .routing import LLMClient, acompletion, aembedding, aresponse, configure, reset
from .types import (
    ChatMessageParam,
    Choice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCallOutputParam,
    FunctionDefParam,
    ResponseOutputItem,
    ResponseOutputText,
    ResponseResult,
    ToolCall,
    ToolCallFunction,
    ToolDefParam,
    Usage,
)

__all__ = [
    # Chat functions
    "chat",
    # Functions
    "acompletion",
    "aembedding",
    "aresponse",
    "configure",
    "reset",
    "should_retry",
    # Client
    "LLMClient",
    # Protocols
    "CompletionProvider",
    "EmbeddingProvider",
    "ResponseProvider",
    # Types — Completion
    "CompletionResponse",
    "Choice",
    "ChatMessageParam",
    "Usage",
    # Types — Tools
    "ToolCall",
    "ToolCallFunction",
    "ToolDefParam",
    "FunctionDefParam",
    "FunctionCallOutputParam",
    # Types — Embedding
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Types — Response
    "ResponseResult",
    "ResponseOutputText",
    "ResponseOutputFunctionCall",
    "ResponseOutputItem",
    # Errors
    "LLMError",
    "AuthenticationError",
    "BadRequestError",
    "RateLimitError",
    "ServerError",
    "LLMTimeoutError",
    "UnsupportedOperationError",
    "ProviderNotAvailableError",
]
