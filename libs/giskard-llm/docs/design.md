# Design Decisions

## Type Conventions

Input types and output types use different base classes:

- **Input types** (`TypedDict`, names end with `Param`): `ChatMessageParam`, `ToolDefParam`, `FunctionDefParam`, `FunctionCallOutputParam`. These are constructed by users or framework code and passed to provider methods. `TypedDict` is lightweight and supports dict literal syntax (`{"role": "user", "content": "hello"}`).

- **Output types** (Pydantic `_BaseModel`): `CompletionResponse`, `Choice`, `ChoiceMessage`, `ToolCall`, `ToolCallFunction`, `EmbeddingResponse`, `ResponseResult`, etc. These are constructed by provider implementations when parsing API responses. Pydantic provides attribute access (`resp.choices[0].message.content`), `.model_dump()` for serialization, and the `_BaseModel` base class defaults `model_dump(exclude_none=True)`.

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

`ToolCallFunction.arguments` is `str` (JSON). This matches the OpenAI SDK convention and ensures wire-compatible round-trips: when `model_dump()` is called on a `ToolCall` and the result is fed back as a message, the `arguments` field is already a JSON string that APIs accept directly. Providers that receive parsed dicts from their SDK (Anthropic `block.input`, Google `fc.args`) serialize them to JSON strings during response normalization.
