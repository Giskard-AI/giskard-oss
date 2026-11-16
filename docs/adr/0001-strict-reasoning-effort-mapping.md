# Strict 1:1 `reasoning_effort` Mapping

Raise `BadRequestError(400)` before calling the provider SDK whenever a caller passes a canonical `reasoning_effort` level that the target provider API does not expose directly, rather than silently clamping or rewriting it.

## Considered Options

- **Silent clamping / rewriting**: e.g., downgrading `minimal` to `low` on Anthropic, `xhigh`/`max` to `high` on Google, or `none` to `thinking_budget=0` on Gemini. Rejected because it hides benchmark and evaluation drift across effort tiers and breaks on Gemini 4 models that forbid `thinking_budget`.
- **Strict 1:1 mapping (chosen)**: normalize case (`"HIGH"` -> `"high"`), translate levels the provider supports natively, and fail fast with `BadRequestError(400)` on unsupported levels or conflicts with provider escape hatches.

## Consequences

- Callers running cross-provider sweeps must pick levels supported by the target provider (all 7 on OpenAI/Azure, `low`–`max` on Anthropic, `minimal`–`high` on Google) or handle `BadRequestError(400)`.
- Legacy budget-based Gemini 2.5 control (`thinking_budget=0`) remains available explicitly through the whitelisted `thinking_config` escape hatch without risking 400s on newer models.
