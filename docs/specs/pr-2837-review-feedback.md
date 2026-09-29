# PR 2837 review feedback

## Problem

Two regressions remain at the current PR head:

1. A check subclass that declares a class-level `generator` default loses that
   default during the legacy `generator` to `judge` migration. Its serialized
   form can also contain both fields and fail to round-trip.
2. `TypeSafeSOM.predict()` reduces each `ChatMessage` to `role` and `text`,
   discarding structured tool calls and tool-result associations.

## Requirements

- Preserve subclass-configured generator defaults as an `LLMChatJudge` unless
  an instance explicitly supplies a judge or generator.
- Serialize only the canonical `judge` representation and retain model
  validation round-trips.
- Send complete JSON-compatible `ChatMessage` payloads to TypeSafe, including
  tool call names, arguments, IDs, and matching tool result IDs.
- Add focused regression tests for both behaviors without changing unrelated
  APIs.
