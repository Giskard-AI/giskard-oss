# Auto-Inject Adaptive Thinking on Anthropic for `reasoning_effort`

When `reasoning_effort` is set on an Anthropic completion call and the caller omits `thinking`, `AnthropicChatTranslator` injects `thinking={"type": "adaptive"}` alongside `output_config["effort"]`.

## Considered Options

- **Set `output_config.effort` only**: Rejected because live testing on Claude 5 showed the model emits no thinking blocks unless `thinking` is enabled, and rejects `thinking={"type": "enabled"}` with HTTP 400—defeating cross-provider portability of `reasoning_effort`.
- **Auto-inject `thinking={"type": "adaptive"}` when `thinking` is omitted (chosen)**: makes canonical `reasoning_effort` work out of the box on Anthropic while still respecting an explicit `thinking` escape hatch when provided.

## Consequences

- Passing `reasoning_effort` alone to Anthropic enables adaptive thinking automatically and merges `output_config["effort"]` with any `output_config["format"]` produced by `response_format`.
- Callers targeting older Claude models that require manual token budgets can pass `thinking={"type": "enabled", "budget_tokens": ...}` explicitly alongside `reasoning_effort`.
