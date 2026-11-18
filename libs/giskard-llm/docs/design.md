# Design Decisions

## Type Conventions

Input types and output types use different base classes:

- **Input types** (`TypedDict`, names end with `Param`): `ChatMessageParam`, `ToolDefParam`, `FunctionDefParam`, `FunctionCallOutputParam`. These are constructed by users or framework code and passed to provider methods. `TypedDict` is lightweight and supports dict literal syntax (`{"role": "user", "content": "hello"}`).

- **Output types** (Pydantic `_BaseModel`): `CompletionResponse`, `Choice`, `AssistantMessage`, `ToolCall`, `ToolCallFunction`, `EmbeddingResponse`, `ResponseResult`, etc. These are constructed by provider implementations when parsing API responses. Pydantic provides attribute access (`resp.choices[0].message.content`), `.model_dump()` for serialization, and the `_BaseModel` base class defaults `model_dump(exclude_none=True)`.

## Which API format we follow

Public inputs and outputs of `giskard.llm` follow the OpenAI API reference. For one method, the object you pass in and the object you get back are the same schema.

| Method | Public input and output | Reference |
|---|---|---|
| `acompletion` | Chat Completions messages in, Chat Completions message out (`ChatMessageParam` / `AssistantMessage`) | [Chat Completions](https://developers.openai.com/api/reference/resources/chat) |
| `aresponse` | Responses items in, Responses items out (`ResponseInputItemParam` / `ResponseResult`) | [Responses](https://developers.openai.com/api/reference/resources/responses) |

Requests follow OpenAI too: `reasoning_effort` is the OpenAI string (ADR-0001, ADR-0002).

### Reasoning: one type, `ReasoningContent` (ADR-0003)

Chat Completions defines no reasoning field, so reasoning has its own convention. It is the same on both methods:

- **Shape.** The OpenAI Responses `reasoning` item, plus two fields the [Open Responses](https://www.openresponses.org) spec lets implementations add:
  - `type: "reasoning"`
  - `id: str | None`, never invented
  - `summary` of `summary_text`
  - `content` of `reasoning_text`
  - `encrypted_content`
  - `status`
  - `signature`
  - `provider` (the originating provider)
- **Where it lives.**
  - In chat: an ordered part of `AssistantMessage.content`, next to `TextContent` and `RefusalContent`.
  - In `aresponse`: an input/output item.
- **`signature` vs `encrypted_content`.**
  - `signature` authenticates visible `content`: Anthropic `thinking.signature`, or the base64 Gemini thought signature.
  - `encrypted_content` replaces content the caller cannot see: OpenAI encrypted reasoning, or Anthropic `redacted_thinking.data`.
- **Provenance.** `provider` is the giskard provider id the call was routed through (`openai`, `azure`, `azure_ai`, `anthropic`, `google`). A serializer replays reasoning only when the tag matches its provider exactly, and drops it silently otherwise. Account-level or region-level rejections (Anthropic account-bound signatures, Gemini API vs Vertex) surface as the provider's 400. Routes of the same family are not interchangeable either: an `openai` item sent to `azure` or `azure_ai` is dropped, so switching routes mid-conversation loses reasoning. Revisit only if a live test shows `encrypted_content` decrypts across them.
- **Order.** `content` keeps reasoning and text parts in the order the provider returned them. `tool_calls` is still a sibling list, so order between reasoning and tool calls is not stored. The Anthropic serializer emits `content` parts in stored order, then `tool_calls`. That reproduces the original turn only when every thinking block came before the first tool call. When an Anthropic response has a thinking block after a `tool_use`, the translator reports it through `unsupported` as `thinking_after_tool_use` (raises unless `ignore_unsupported_content`, which drops the block). Whether Claude produces that shape with client tools is not yet verified; a live test should try to trigger it. When consecutive assistant messages are merged into one Anthropic turn, thinking blocks are moved to the front of the merged turn (stable sort), because Anthropic requires a turn to lead with thinking. Reasoning parts inside tool-result content are dropped: Anthropic only accepts thinking in assistant turns.
- **Type.** `ReasoningContent` replaces the unreleased `ResponseReasoningItem` (renamed, `id` made optional). Any check that an `openai` item has an `id` moves from the model to the OpenAI Responses translator, which rejects a replayed reasoning item without `id` with `BadRequestError` before serializing the request.

Who replays:

| Path | Replays own reasoning | Notes |
|---|---|---|
| Anthropic chat | yes: `thinking` (text + `signature`), `redacted_thinking` (`encrypted_content`) | required for tool loops with thinking |
| OpenAI / Azure Responses | yes: `id`, `summary`, `encrypted_content` | |
| Google Responses | yes: thought step `summary` + `signature` | |
| Google chat | no | thought parts are stored in `content` (never in `.text`) and skipped on the way out; tool-call signatures live on `ToolCall.thought_signature` (#2965) |
| OpenAI chat (incl. OpenAI-compatible servers) | no | Chat Completions has no field; reasoning parts are stripped from `content` |

`ReasoningContent` is a superset of every provider's shape, so what goes on the wire is not the public object. Serializer rules:

- **One block, one part.** Each provider reasoning block becomes exactly one `ReasoningContent`. Blocks are never merged, or every signature but one is lost.
- **`provider` is never sent.** It is a Giskard field.
- **OpenAI / Azure Responses** send `id`, `summary` and `encrypted_content` only. `content` and `signature` stay off the request (Open Responses requires reasoning `content` to be null on input).
- **Google Responses** maps `signature` onto the thought step signature. `encrypted_content` is never written there.
- **Anthropic** sends a `thinking` block (text + `signature`) for each part with a `signature`, including when the text is empty: empty means `display: "omitted"`, not "no block". A part with `encrypted_content` becomes `redacted_thinking`.
- **Replay reads `content` parts only.** The deprecated `reasoning` string is never a replay source, including when `content` is a plain string.
- **Answer text skips reasoning by type.** `_extract_text` (and so `.text` / `.transcript`) filters out `ReasoningContent` with an isinstance check. It must not rely on a missing `.text`, which raises, nor give the part a `.text`, which would leak thoughts into answers.

OpenAI-compatible servers return reasoning as `reasoning`, `reasoning_content` (vLLM, DeepSeek) or `reasoning_details` (OpenRouter). `OpenAIChatTranslator` converts all three to `ReasoningContent` tagged with the routed provider id and placed before the answer text. `reasoning_details` maps item by item (`reasoning.text` to `content` + `signature`, `reasoning.summary` to `summary`, `reasoning.encrypted` to `encrypted_content`, keeping `id`); when it is present, the plain `reasoning` string is discarded, since it duplicates the `reasoning.text` items. It ignores OpenRouter's `format` and `index`. The raw keys are removed after conversion, so the public type never carries `reasoning_details`; the OpenRouter wire key is still read on the way in.

`AssistantMessage.reasoning: str` is released (1.0.x) and stays through 1.x as a deprecated convenience: it is filled from the parts' `reasoning_text` only, never replayed, and removed in 2.0. A message whose reasoning is only `summary` or `encrypted_content` has `reasoning is None`. Do not add OpenRouter fields (`reasoning_details`, `format` tags) or a second reasoning type to the public types.

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

In `giskard-agents`, `GenerationParams` uses `extra="forbid"` and `BaseGenerator.with_params(**kwargs)` validates updates while preserving `model_fields_set`. Because `GiskardLLMGenerator` and `LiteLLMGenerator` serialize with `exclude_unset=True`, the default `temperature=1.0` is omitted unless explicitly set, avoiding spurious 400s on reasoning models that forbid non-default `temperature`.

## Reasoning Output & Token Accounting

Public reasoning is `ReasoningContent`, see [Reasoning: one type](#reasoning-one-type-reasoningcontent-adr-0003).

- **Answer isolation**: `ReasoningContent` parts and the deprecated `AssistantMessage.reasoning` never contribute to `.text`, `.transcript`, or `.output_text`. Serializers send reasoning only to its originating provider.
- **Google thought parts**: `GoogleChatTranslator` turns every `Part.thought == True` part into a `ReasoningContent` part tagged `google`, never `TextContent`, including thought parts with empty or missing text. A thought signature is base64-encoded into `signature`. Google chat does not replay these parts.
- **Reasoning tokens**: `Usage.reasoning_tokens: int | None = None` captures reasoning token counts from OpenAI Chat (`completion_tokens_details.reasoning_tokens`), OpenAI Responses (`output_tokens_details.reasoning_tokens`), Anthropic Messages (`output_tokens_details.thinking_tokens`), Google Chat (`usage_metadata.thoughts_token_count`, which remains included in `output_tokens`), and Google Interactions (`usage.total_thought_tokens`).
