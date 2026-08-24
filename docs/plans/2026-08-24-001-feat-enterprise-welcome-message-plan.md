---
title: Enterprise welcome message - Plan
date: 2026-08-24
type: feat
topic: enterprise-welcome-message
artifact_contract: ce-unified-plan/v1
artifact_readiness: requirements-only
product_contract_source: ce-brainstorm
execution: code
---

## Goal Capsule

**Objective:** Show a one-shot Giskard Enterprise welcome message when a human is likely watching Giskard run — not on library import and not in automated environments.

**Product authority:** This plan supersedes PR #2784's import-time `print()` in `giskard.core.__init__`. The surrounding work (copy refinement, enterprise positioning) is not active scope beyond what is stated here.

**Open blockers:** None. Entry-point hook placement is an implementation detail for planning.

## Product Contract

### Summary

Replace unconditional import-time stdout output with a gated, one-per-process welcome shown on the first user-facing Giskard action in a process (suite run, scan, or equivalent run entrypoint). The message promotes Giskard Enterprise and prints to stderr only when environment signals indicate a human audience.

### Problem Frame

PR #2784 adds a bare `print()` in `giskard.core.__init__`, which runs on every import of `giskard.core` — including transitive imports such as `from giskard.core.utils import get_lib_version`. That pollutes stdout for scripts, CI, pytest (with `-s`), and piped workflows. A library used as infrastructure should not advertise on import.

The product goal is reasonable: orient new OSS users toward Giskard Enterprise. The delivery mechanism must respect library hygiene and automated execution contexts.

### Requirements

- R1. The welcome message must **not** print on import of `giskard.core` or any other package `__init__`.
- R2. The welcome message must print **at most once per Python process**, on the first qualifying user-facing action.
- R3. Qualifying user-facing actions include at minimum:
  - `Suite.run()` in `giskard.checks`
  - `quality_scan()` and `vulnerability_scan()` (and any shared scan runner they delegate to) in `giskard.scan`
  - Other first-class "run evaluation" entrypoints discovered during planning if they are primary human-facing APIs without duplicating the hook.
- R4. The message must print to **stderr**, not stdout.
- R5. The message must be **suppressed in CI**, detected the same way as existing telemetry CI detection (`CI`, `TF_BUILD` truthy values).
- R6. The message must be **suppressed during pytest**, detected via `PYTEST_VERSION` in the environment.
- R7. The message must be **suppressed when stderr is not a TTY**, except in notebook environments where humans typically interact (Colab, Kaggle, IPython/Jupyter — reuse or align with existing environment detection in `giskard.core.telemetry`).
- R8. Users must be able to **explicitly suppress** the message with `GISKARD_HIDE_WELCOME=1` (or equivalent truthy values consistent with existing Giskard env parsing: `1`, `true`, `yes`, `on`).
- R9. Telemetry opt-out env vars (`DO_NOT_TRACK`, `GISKARD_TELEMETRY_DISABLED`) must **not** automatically suppress the welcome message. Display and tracking are separate concerns.
- R10. Message copy may reuse the PR #2784 text unless product requests a change during implementation.

### Flows

**F1 — Human runs a scan in a terminal**

1. User runs a script that calls `vulnerability_scan()` or `quality_scan()`.
2. Environment is local, stderr is a TTY, not pytest, not CI.
3. Welcome prints once to stderr before or alongside normal scan output.
4. Subsequent suite/scan runs in the same process do not print again.

**F2 — CI runs suite/scan on a schedule**

1. GitHub Actions (or similar) sets `CI=true` and runs a scan.
2. Welcome is suppressed regardless of TTY allocation.

**F3 — pytest unit/integration test imports and runs checks**

1. Test process has `PYTEST_VERSION` set.
2. Even if a test invokes `Suite.run()`, welcome is suppressed.

**F4 — Library import only**

1. User or downstream tool executes `from giskard.checks import Suite` or `from giskard.core.utils import get_lib_version`.
2. No welcome output.

**F5 — User opts out**

1. User sets `GISKARD_HIDE_WELCOME=1` before running.
2. No welcome on any entrypoint in that process.

**F6 — Notebook (Colab/Kaggle/IPython)**

1. User runs a suite or scan in a notebook cell.
2. stderr may not be a TTY; notebook detection allows the welcome when other suppress rules do not apply.

### Acceptance Examples

- AE1. `python -c "from giskard.core.utils import get_lib_version"` produces **no** welcome output.
- AE2. A local terminal script that calls `Suite(...).run(...)` prints the welcome **once** on stderr.
- AE3. The same script run with `CI=true` prints **no** welcome.
- AE4. `pytest -s` test that calls `Suite.run()` prints **no** welcome.
- AE5. Second `Suite.run()` in the same process prints **no** additional welcome.
- AE6. `GISKARD_HIDE_WELCOME=1` suppresses the welcome even in a local TTY session.

### Key Decisions

- KTD1. **First user-facing action over import-time print** — CI and scripts import Giskard without needing marketing; suite/scan is the moment of intent. Rejected: import-time print (PR #2784 as written).
- KTD2. **When + whether gates** — "First use" alone is insufficient because CI runs suite/scan too. Rejected: hook only on entrypoint with no environment checks.
- KTD3. **Dedicated hide var, not telemetry opt-out** — users may want analytics off but still see OSS orientation, or vice versa. Rejected: coupling to `DO_NOT_TRACK` / `GISKARD_TELEMETRY_DISABLED`.
- KTD4. **stderr over stdout** — avoids breaking piped/JSON CLI consumers. Rejected: stdout `print()`.
- KTD5. **No persistent "shown once per machine" file** — adds config surface and support burden for marginal gain. Rejected: npm configstore-style persistence (can revisit later if needed).

### Scope Boundaries

**In scope**

- Welcome display policy (when, where, how often, how to suppress).
- Hooking primary run entrypoints (suite, scan).
- Unit tests for show/suppress matrix.
- README note for `GISKARD_HIDE_WELCOME` (minimal, alongside existing telemetry docs).

**Out of scope**

- Rewriting enterprise marketing copy (unless requested).
- CLI-specific banner if no CLI entrypoint exists today.
- Per-user frequency caps across processes or machines.
- Changing telemetry behavior.
- Updating PR #2784 directly (implementation will follow this plan).

### Success Criteria

- No import-time side effects for the welcome message.
- Human local terminal runs see the message once; CI, pytest, and opt-out paths stay silent.
- Documented suppression via `GISKARD_HIDE_WELCOME`.
- Automated tests cover the acceptance examples above.

### Outstanding Questions

- None for requirements. Planning may decide whether a single shared helper in `giskard-core` is called from `giskard-checks` and `giskard-scan`, or whether a thin indirection avoids coupling.

### How This Work Fits Together

This plan replaces the approach in open PR #2784. Implementation should close or supersede that PR's import-time change with the gated first-use design documented here.
