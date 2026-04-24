# Design Decisions

## Type Conventions

Input types and output types use different base classes:

- **Input types** (`TypedDict`, names end with `Param`): `ChatMessageParam`, `ToolDefParam`, `FunctionDefParam`, `FunctionCallOutputParam`. These are constructed by users or framework code and passed to provider methods. `TypedDict` is lightweight and supports dict literal syntax (`{"role": "user", "content": "hello"}`).

- **Output types** (Pydantic `_BaseModel`): `CompletionResponse`, `Choice`, `AssistantMessage`, `ToolCall`, `ToolCallFunction`, `EmbeddingResponse`, `ResponseResult`, etc. These are constructed by provider implementations when parsing API responses. Pydantic provides attribute access (`resp.choices[0].message.content`), `.model_dump()` for serialization, and the `_BaseModel` base class defaults `model_dump(exclude_none=True)`.

## Tool Definition Format

The Chat Completions API (OpenAI, Azure, Anthropic, Google via `generateContent`) uses a **nested** tool format:

```python
{"type": "function", "function": {"name": "add", "description": "...", "parameters": {...}}}
```

The Responses API (OpenAI) and Interactions API (Google) use a **flat** tool format:

```python
{"type": "function", "name": "add", "description": "...", "parameters": {...}}
```

The library accepts `ToolDefParam` (nested Chat Completions format) as the single public input type for all methods. Each provider's `respond()` method flattens `ToolDefParam` to the flat format before calling the underlying API.

## Tool Result Format

When feeding back function call results to `respond()`, the canonical format is `FunctionCallOutputParam` (OpenAI-like):

```python
{"type": "function_call_output", "call_id": "...", "name": "add", "output": "7"}
```

The `name` field is required because Google's Interactions API needs it. OpenAI's Responses API ignores it.

`GoogleProvider.respond()` normalizes `FunctionCallOutputParam` items to the Google-native `function_result` format internally. OpenAI passes items through as-is.

## `ToolCallFunction.arguments` Type

`ToolCallFunction.arguments` is `ArgumentDict` (`dict[str, object]`). A `BeforeValidator` automatically parses JSON strings on input, so callers may provide either a raw dict or a JSON string — both are stored as a dict. This makes tool arguments safe to inspect and pass to function calls without a separate parsing step.

Translators that need to send arguments back to an API (e.g. Anthropic `tool_use` blocks) use `serialize_arguments()` from `giskard.llm.utils` to convert the dict to a JSON string. The `serialize_arguments` / `deserialize_arguments` helpers both accept `dict | str` for defensive handling.
