# Reasoning Is an Ordered Content Part Shared by Chat and Responses

OpenAI Chat Completions has no field for reasoning on assistant messages, so "follow OpenAI" cannot describe chat history. Yet Anthropic tool loops with thinking must replay signed `thinking` and `redacted_thinking` blocks unchanged and in order. We carry reasoning as one public type, `ReasoningContent`, that appears both as an ordered part of `AssistantMessage.content` (next to `TextContent` and `RefusalContent`) and as a Responses input/output item. Its fields are the OpenAI Responses reasoning item's (`id`, `summary`, `content`, `encrypted_content`) plus two implementation-specific fields that the Open Responses spec allows: `signature` and the Originating Provider (the giskard provider id). Reasoning is replayed only on an exact provider match and silently dropped otherwise.

## Considered Options

- **Separate `reasoning_details` list (OpenRouter shape)**: rejected. It is a different shape from the Responses item, so there would be two conventions. It sits beside `content`, so order relative to text and tool calls is lost. Only gateways use it, and giskard-llm never sends it to OpenRouter anyway.
- **`aresponse` as the single API (v2)**: rejected for now. Anthropic tool loops through `acompletion` would still need chat replay, so it does not remove this decision.
- **Content part reusing the Responses item (chosen)**: this is how Anthropic and Gemini natively represent reasoning, and how Vercel AI SDK, pydantic-ai and LangChain v1 do it too. Order relative to answer text is stored. Order relative to tool calls is not, because `tool_calls` stays a sibling list; the Anthropic serializer needs an explicit merge rule (see `libs/giskard-llm/docs/design.md`). It is additive in 1.x.

## Consequences

- `signature` authenticates visible `content` (Anthropic thinking, Gemini thought). `encrypted_content` replaces content the caller cannot see (OpenAI encrypted reasoning, Anthropic redacted thinking). Keeping them apart avoids sending a signature as redacted `data` when `display:"omitted"` yields empty thinking text.
- `id` is optional and never invented: only OpenAI-tagged reasoning reaches OpenAI, and it always has one.
- `AssistantMessage.reasoning: str` stays through 1.x as a derived, never-replayed, deprecated field, to be removed in 2.0.
- Matching is by exact route, not model family: `openai` reasoning is dropped on `azure` / `azure_ai`. Whether `encrypted_content` decrypts across those routes is unverified, and a blob that doesn't would turn a silent drop into a 400.
- `tool_calls` stays a sibling list, so an Anthropic turn with thinking after a `tool_use` cannot be stored in order. The translator reports it as unsupported instead of reordering it into a 400 on replay. Moving tool calls into `content` was rejected: it would end the Chat Completions shape of `AssistantMessage`.
- `ReasoningContent` replaces the unreleased `ResponseReasoningItem`. One provider block maps to one part; serializers send only the fields their provider accepts, and never `provider` (rules in `libs/giskard-llm/docs/design.md`).
- Serializers that send no reasoning (OpenAI chat) must strip reasoning parts from `content`. `text` / `transcript` must skip them by type: `_extract_text` reads `.text` on every part, so a part without one crashes it and a part with one leaks thoughts.
- giskard-llm 1.0.x cannot load traces containing a reasoning part (closed `CompletionContent` union). In-repo readers of raw `turn.content` (giskard-scan `integrations/_shared.py`, `lidar/_target.py`) see reasoning parts too and must skip them. This is accepted as forward-incompatibility in a minor release.
- Gemini thought signatures on text parts and tool calls are out of scope (#2965).
- Google chat keeps not replaying its own thought parts. Gemini requires signatures on function calls, which `ToolCall` carries, not on thoughts.
- Reasoning from OpenAI-compatible servers (`reasoning`, `reasoning_content`, OpenRouter `reasoning_details`) is tagged with the routed provider id. OpenRouter's `format` is ignored. Nothing is replayed through OpenAI chat, so Claude tool loops with thinking through an OpenRouter `base_url` get no replay, the same as before.
