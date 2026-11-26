# Giskard OSS

Unified language for Giskard's LLM routing, agent workflows, evaluation checks, and safety scanning packages.

## Language

### Reasoning & Generation Parameters

**Reasoning Effort**:
The canonical seven-level setting (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) specifying how much internal deliberation a model performs before answering.
_Avoid_: `thinking_level`, `thinking_budget`, `effort` (outside provider wire payloads)

**Reasoning Content**:
Internal model deliberation returned alongside an assistant turn, kept strictly separate from visible answer text so checks and judges never evaluate private thoughts.
_Avoid_: `thinking` (as a Giskard public field), `chain_of_thought`

**Reasoning Detail**:
A structured, provider-tagged reasoning artifact (`reasoning.text`, `reasoning.summary`, or `reasoning.encrypted`) attached to an assistant turn for verifiable replay to the originating provider.
_Avoid_: raw SDK block dicts

**Reasoning Tokens**:
Generated output tokens consumed by internal deliberation, included in billed output tokens and tracked independently of visible answer tokens.
_Avoid_: `thinking_tokens`, `thoughts_token_count` (outside provider translators)

**Provider Escape Hatch**:
A whitelisted provider-native parameter (`thinking`, `thinking_config`, `output_config`, `reasoning`, `generation_config`) forwarded by a translator for provider controls that have no canonical equivalent.
_Avoid_: arbitrary kwargs, unvalidated `extra_body`

**Generation Params**:
The unified per-generator and per-call configuration (`GenerationParams`) merged from base defaults and workflow overrides, serialized with unset fields omitted.
_Avoid_: untyped parameter dicts on generators
