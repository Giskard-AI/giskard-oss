# Plan: SOM judge groundwork — shared trace, per-check questions

## Goal

Align judge prompts with SOM’s natural shape so future batching is possible:

- **Input (shared):** conversation `trace`
- **Questions (per check):** rendered rubric + check variables (no trace, no `_instr_output`)

This PR does **not** implement multi-question grouping/batching; it only lays that ground.

## Template contract

Bundled judge `.j2` files:

1. Rubric + check-specific markers (`rule`, `answer`, `context`, `question`, `output`, …) are always rendered (no `include_rubric`).
2. Only the conversation / history block is gated with `{% if include_trace | default(true) %}`.
3. Output schema stays behind `{% if _instr_output is defined %}`.
4. Remove `include_evidence` / `include_rubric`.

Checks that already embed answer/context (groundedness, contradiction) keep those always-on; they do not need a TRACE section in the template. They still pass `trace` in `get_inputs` for the SOM messages path.

## SOMJudge

1. Render prompt once with `include_trace=False` → SOM `question` (fallback `_DEFAULT_SOM_QUESTION` if empty).
2. Build messages from `inputs["trace"]` via `{{ trace | fence }}` (not a second full prompt render).
3. Fail closed when `trace` is missing (except a plain `ChatMessage` prompt, which is the message list).
4. Extract small helpers (`_som_question_from_prompt`, `_som_messages_from_trace`) documenting that many questions can later share one trace input.

## Check inputs

- `Groundedness` / `Contradiction` / `AnswerRelevance`: always include `"trace": trace` in `get_inputs`.
- Conformity / toxicity already pass `trace`.

## Docs / tests

- README: document `include_trace` + SOM question/messages split; note batching is future work.
- Replace dual-render / empty-evidence tests with the new contract.
