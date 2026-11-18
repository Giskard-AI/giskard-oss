# Giskard OSS

Unified language for Giskard's LLM routing, agent workflows, evaluation checks, and safety scanning packages.

## Language

### Reasoning & Generation Parameters

**Reasoning Effort**:
The canonical seven-level setting (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) specifying how much internal deliberation a model performs before answering.
_Avoid_: `thinking_level`, `thinking_budget`, `effort` (outside provider wire payloads)

**Reasoning Content**:
Internal model deliberation returned as an ordered part of an assistant turn, possibly as a summary, signed, or encrypted, and tagged with its Originating Provider. Kept strictly separate from visible answer text so checks and judges never evaluate private thoughts. The same concept on chat history and on the Responses API.
_Avoid_: `thinking` (as a Giskard public field), `chain_of_thought`, Reasoning Detail, `reasoning_details`, reasoning item

**Originating Provider**:
The provider route, as Giskard names it, that a piece of reasoning came back through. It is the route, not the model vendor: Claude reasoning returned through an OpenAI-compatible gateway originates from that gateway's route. Reasoning is replayed only to its originating provider; sent anywhere else, it is silently dropped.
_Avoid_: `format`, source, vendor, model provider

**Reasoning Tokens**:
Generated output tokens consumed by internal deliberation, included in billed output tokens and tracked independently of visible answer tokens.
_Avoid_: `thinking_tokens`, `thoughts_token_count` (outside provider translators)

**Provider Escape Hatch**:
A whitelisted provider-native parameter (`thinking`, `thinking_config`, `output_config`, `reasoning`, `generation_config`) forwarded by a translator for provider controls that have no canonical equivalent.
_Avoid_: arbitrary kwargs, unvalidated `extra_body`

**Generation Params**:
The unified per-generator and per-call configuration (`GenerationParams`) merged from base defaults and workflow overrides, serialized with unset fields omitted.
_Avoid_: untyped parameter dicts on generators
