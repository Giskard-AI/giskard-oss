# Design Decisions

## Type Conventions

Input types and output types use different base classes:

- **Input types** (`TypedDict`, names end with `Param`): `ChatMessageParam`, `ToolDefParam`, `FunctionDefParam`, `FunctionCallOutputParam`. These are constructed by users or framework code and passed to provider methods. `TypedDict` is lightweight and supports dict literal syntax (`{"role": "user", "content": "hello"}`).

- **Output types** (Pydantic `_BaseModel`): `CompletionResponse`, `Choice`, `AssistantMessage`, `ToolCall`, `ToolCallFunction`, `EmbeddingResponse`, `ResponseResult`, etc. These are constructed by provider implementations when parsing API responses. Pydantic provides attribute access (`resp.choices[0].message.content`), `.model_dump()` for serialization, and the `_BaseModel` base class defaults `model_dump(exclude_none=True)`.

## Tool Definition Format

The Chat Completions API (OpenAI, Azure, Anthropic, Google via `generateContent`) uses a **nested** tool format:

```python
{
    "type": "function",
    "function": {"name": "add", "description": "...", "parameters": {...}},
}
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

## Reasoning Effort & Provider Escape Hatches

`acompletion`, `aresponse`, and `giskard-agents` (`GenerationParams`) share a canonical `ReasoningEffort` literal (`"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`), case-normalized on input.

Translators enforce a **strict 1:1 mapping** (`BadRequestError(400)` on unsupported levels) and merge `reasoning_effort` cleanly with co-located provider parameters rather than overwriting them:

- **OpenAI / Azure (`complete`)**: passes `reasoning_effort` directly (all 7 levels).
- **OpenAI / Azure (`respond`)**: merges `effort` into `reasoning` (`all 7` levels).
- **Anthropic (`complete`)**: merges `effort` into `output_config` alongside `output_config["format"]` from `response_format` (`low`..`max`; rejects `none`, `minimal`), and sets `thinking={"type": "adaptive"}` when `thinking` is omitted.
- **Google (`complete`)**: merges uppercased `thinking_level` into `config.thinking_config` (`minimal`..`high`; rejects `none`, `xhigh`, `max`).
- **Google (`respond`)**: merges lowercased `thinking_level` into `generation_config` alongside `temperature` (`minimal`..`high`; rejects `none`, `xhigh`, `max`).

Provider-native escape hatches (`reasoning`, `thinking`, `output_config`, `thinking_config`, `generation_config`) are whitelisted in `KNOWN_*_PARAMS`. Passing both `reasoning_effort` and the conflicting native effort/budget key raises `BadRequestError(400)`.

`reasoning_effort` only controls how much the model thinks; it never changes what is returned (no implicit rewriting, see ADR 0001). On Gemini (`complete`) the thought text is only returned when requested, so `AssistantMessage.reasoning` stays empty — even though reasoning tokens are spent and reported in `Usage.reasoning_tokens` — unless the caller opts in with the escape hatch, which merges with `reasoning_effort`: `thinking_config={"include_thoughts": True}`.

In `giskard-agents`, `GenerationParams` uses `extra="forbid"` and `BaseGenerator.with_params(**kwargs)` validates updates while preserving `model_fields_set`. Because `GiskardLLMGenerator` and `LiteLLMGenerator` serialize with `exclude_unset=True`, the default `temperature=1.0` is omitted unless explicitly set, avoiding spurious 400s on reasoning models that forbid non-default `temperature`.

## Reasoning Output & Token Accounting

- **Answer isolation**: `AssistantMessage.reasoning`, `AssistantMessage.reasoning_details`, and `ResponseReasoningItem` never contribute to `.text`, `.transcript`, or `.output_text`. Outbound serializers strip reasoning via `_BaseModel._wire_excluded`, except provider-tagged details explicitly required for tool-loop continuity (e.g. signed `anthropic-claude-v1` blocks and `ResponseReasoningItem`).
- **Google thought parts**: `GoogleChatTranslator` routes `Part.thought == True` parts to `AssistantMessage.reasoning` rather than `TextContent`.
- **Reasoning tokens**: `Usage.reasoning_tokens: int | None = None` captures reasoning token counts from OpenAI Chat (`completion_tokens_details.reasoning_tokens`), OpenAI Responses (`output_tokens_details.reasoning_tokens`), Anthropic Messages (`output_tokens_details.thinking_tokens`), Google Chat (`usage_metadata.thoughts_token_count`, which remains included in `output_tokens`), and Google Interactions (`usage.total_thought_tokens`).
