# Design: Eval checkpoint / resume (parallel-safe)

**Date:** 2026-07-29
**Status:** Implemented (branch `feat/eval-checkpoint-resume`)
**Package focus:** `giskard-checks` (store + `Suite.run`), `giskard-scan` (`generate_suite`)

## Problem

Long-running evals lose all progress on interrupt. Today:

- `Suite.run` keeps `ScenarioResult`s in memory until every scenario finishes, then returns one `SuiteResult`.
- `generate_suite` runs generators concurrently via `TaskGroup` and only returns a `Suite` after all generators complete.
- Progress UI updates live; nothing is persisted. Ctrl+C, OOM, laptop sleep, or CI timeout discards completed work.

This must work for **parallel** generation and **parallel** suite execution (out-of-order completion, concurrent writers).

## Goals

1. **Durable mid-run backups** — completed units survive process death (best-effort for `SIGKILL`).
2. **Resume** — re-run skips completed units and merges with newly finished ones.
3. **Partial artifacts** — interrupted runs still yield a usable partial suite / suite result for reporting.
4. **Parallel-safe** — concurrent appends must not corrupt the store; resume keys are stable IDs, not completion order.
5. **Cover both phases** — suite **generation** and suite **execution**.

## Non-goals (v1)

- Distributed / multi-machine orchestration.
- Cross-run cache reuse when config fingerprints differ.
- Step- or item-level checkpointing (store schema must allow it later).
- Always-on checkpointing (opt-in only).
- SQLite or remote backends.

## Decisions (locked)

| Topic | Choice |
|---|---|
| Failure modes | Both resume and partial reports (C) |
| Granularity | Scenario-level first; schema extensible to steps / generation items (D) |
| Opt-in | Auto under `.giskard/checkpoints/<fingerprint>/` with resume default `True`; disable via `checkpoint_dir=False` or `GISKARD_CHECKPOINT=0` |
| Placement | Thin `RunStore` helper + wire into `Suite.run` and `generate_suite` (C) |
| Storage | Append-only JSONL run store (approach 1) |

## Architecture

```text
┌─────────────────┐     append scenario      ┌──────────────────┐
│ generate_suite  │ ───────────────────────► │                  │
│ (TaskGroup)     │                          │  RunStore        │
└─────────────────┘                          │  (JSONL + meta)  │
                                             │                  │
┌─────────────────┐     append result        │  - manifest.json │
│ Suite.run       │ ───────────────────────► │  - events.jsonl  │
│ (serial/parallel│                          │                  │
│  TaskGroup)     │ ◄── load completed IDs ──│                  │
└─────────────────┘                          └──────────────────┘
```

### RunStore

Small helper (prefer `giskard-checks` so both packages can use it without a scan→checks cycle):

- **Open / create** a checkpoint directory.
- **Append** an event record (atomic line write + flush).
- **List completed IDs** by event type.
- **Load** payloads for resume / partial assemble.
- **Run fingerprint** in `manifest.json` so resume refuses mismatched configs (or requires `resume="force"`).

Suggested layout:

```text
{checkpoint_dir}/
  manifest.json          # fingerprint, created_at, phase, schema_version
  events.jsonl           # one JSON object per line
```

Event types (v1):

- `scenario_generated` — payload: stable `id`, scenario dump, generator key, seed metadata
- `scenario_finished` — payload: same `id`, `ScenarioResult` dump

Later (non-v1): `step_finished`, `item_generated`, etc.

### Stable IDs

`Scenario` today has `name` only (not guaranteed unique). On first checkpoint write (or at generation time):

- Assign `checkpoint_id` (UUID or deterministic hash of generator + seed + index + name).
- Prefer storing it in scenario `annotations` (or a dedicated field if we add one later) so run and resume share the same key.

Resume skips tasks whose `checkpoint_id` already has a matching completed event.

### Parallel writing

- **Append-only JSONL**; each record is one line.
- Serialize writers with an `asyncio.Lock` (or a single writer task + queue) so concurrent scenario completions do not interleave bytes within a line.
- `flush` after each append so a crash loses at most in-flight tasks, not already-finished ones.
- Do **not** rewrite a shared mutable `results.json` from many tasks.

### Opt-in API / env

**API (illustrative):**

```python
await generate_suite(..., checkpoint_dir="...", resume=True)
await suite.run(..., checkpoint_dir="...", resume=True)
```

**Env (illustrative):**

- `GISKARD_CHECKPOINT_DIR` — default directory when API arg omitted
- `GISKARD_CHECKPOINT_RESUME=1` — enable resume when dir is set

When neither API nor env sets a dir, behavior is unchanged (no disk I/O).

### Fingerprint

`manifest.json` should include enough to detect incompatible resume, e.g.:

- phase (`generate` | `run`)
- `seed`, `target_mode`, generator class names / versions (generate)
- suite name + ordered scenario `checkpoint_id`s (run)
- library / schema version

Mismatch → clear error unless explicit force.

### Partial report / interrupt

- On normal completion: assemble full `Suite` / `SuiteResult` from memory (or reload from store); optionally leave checkpoint in place.
- On cancel / handled interrupt: assemble **partial** result from store + in-memory finished tasks; re-raise or return depending on existing `return_exception` semantics (document the choice in the plan).
- `SIGKILL` / power loss: last flushed JSONL lines remain; user re-runs with `resume=True`.

### Privacy

Checkpoint files may contain prompts, traces, and model outputs. Document that dirs can hold sensitive data; recommend local paths and gitignoring default locations. No cloud upload in v1.

## Integration points

1. **`giskard.checks.scenarios.suite.Suite.run`** — after each scenario completes (serial loop and parallel `run_scenario`), append `scenario_finished`; on start with `resume`, skip completed IDs and prepend loaded results (preserve suite order in the final list).
2. **`giskard.scan.catalog.generate_suite` / `_generate_scenarios`** — as each generator task completes, append each produced scenario as `scenario_generated`; on resume, skip generators (or scenarios) already present per fingerprint + IDs. v1 may resume at **generator** batch granularity if per-scenario emit inside generators is too invasive; prefer per-scenario when the generator returns a list (write each scenario as it is collected after the generator returns is still “all or nothing” per generator — acceptable for v1 if documented).
3. **Scan helpers** (`vulnerability_scan`, `quality_scan`, `third_party_scan`) — pass through checkpoint kwargs / env so users do not need to drop to raw `Suite`.

### Generation resume nuance (v1)

Generators always run under `TaskGroup`. Ideal: stream scenarios out as produced. Practical v1: when a generator’s `generate_scenario` returns, append all its scenarios, then mark that generator complete. Resume skips finished generators. Finer item-level streaming is a follow-up (schema already allows new event types).

## Testing

- Unit: `RunStore` append / load / lock under concurrent fake writers.
- Unit: resume skips completed scenario IDs; order of final `SuiteResult.results` matches suite order.
- Unit: fingerprint mismatch raises.
- Integration: interrupt mid-`Suite.run(parallel=True)` (cancel tasks), assert partial JSONL + successful resume without re-running completed stubs.
- Integration: interrupt mid-`generate_suite`, resume completes suite without re-calling finished generators.

## Rollout

1. Land `RunStore` + tests in `giskard-checks`.
2. Wire `Suite.run`.
3. Wire `generate_suite` (+ scan passthrough).
4. Docs / examples for API + env; note privacy and resume fingerprint.

## Open follow-ups (not blocking v1)

- Step / check / generation-item events.
- Streaming scenarios from inside long generators.
- Optional compact “snapshot.json” for faster load of large runs.
- Hub upload of checkpoint dirs.
