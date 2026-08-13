# AGENTS.md

giskard-oss — behavioral config for interactive coding assistants with a human in the loop.

## Workflow Orchestration

### 1. Plan Mode Default
– Enter plan mode for ANY non-trivial task (3+ steps or touches multiple libs)
– Brainstorm first (design → spec), then write an implementation plan
– Save specs and plans in the repo's established planning docs. Prefer `docs/specs/` and `docs/plans/`; if the repo already uses another planning directory, follow that convention.
– If something goes sideways, STOP and re-plan immediately

### 2. Subagent Strategy
– Use subagents or equivalent isolated work sessions when the assistant supports them
– Prefer a fresh subagent/session per task; controller curates exactly the context each one needs
– Two-stage review after each task: spec compliance first, then code quality

### 3. Self-Improvement Loop
– After ANY correction: self-document the rule (naming, file headers, examples); only add to AGENTS.md if it cannot be self-documented
– When a correction changes what was implemented vs. what the plan says: update the plan file immediately so reviewers never see a mismatch

### 4. Verification Before Done
– Never mark a task complete without proving it works
– Run: `make format && make check && make test-unit PACKAGE=<affected-lib>`
– Show actual output. Never assume tests pass.
– Ask yourself: "Would a staff engineer approve this?"

### 5. Demand Elegance
– Pause and ask "is there a more elegant solution?"
– Skip for simple fixes — don't over-engineer

### 6. Autonomous Bug Fixing
– When given a bug report with clear scope: just fix it
– No `# type: ignore`, no patched test assertions — fix the root cause

### 7. Commit Hygiene
– Use Conventional Commits for every commit message (for example, `fix(llm): update google interactions translator`)

## Task Management
1. Brainstorm → Spec → Plan — save to the repo's established planning docs, preferably `docs/specs/` and `docs/plans/`
2. Verify Plan — check in before starting
3. Track Progress — mark items complete as you go
4. Explain Changes — high-level summary at each step
5. Document Results — add review section to the plan file
6. Capture Lessons — update AGENTS.md directly after corrections

## Core Principles
– Simplicity First: make every change as simple as possible; prefer deleting lines over adding them
– No Laziness: find root causes; no band-aids, no temporary fixes; senior developer standards
– Minimal Impact: only touch what's necessary; no side effects; no reformatting untouched lines

## Cursor Cloud specific instructions

Non-obvious context for cloud agents (startup refresh is `make install` then `make install-tools`; `uv` is on `PATH` for login shells via `~/.bashrc`).

– This repo is a `uv` workspace monorepo of pure Python libraries (`libs/giskard-*`). There is no server, daemon, or CLI to start — "running" the product means importing the libraries from Python or running `pytest`. Prefer Makefile targets over raw `uv`/`pytest` (`make help`).
– Import namespace is `giskard.<sublib>` (e.g. `import giskard.checks`), not `giskard_checks`.
– Do not run full `make setup` in this VM: its `pre-commit-install` step fails with `Cowardly refusing to install hooks with core.hooksPath set`. That is expected (global `core.hooksPath`) and harmless — use `make install` + `make install-tools` instead; git hooks are not needed to validate work.
– Lint/test/build/run all work without secrets: `make lint`, `make test-unit [PACKAGE=<lib>]`, and importing the libs. Unit tests exclude anything marked `functional`.
– Functional/integration tests are excluded by default and require live LLM provider API keys (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`); `giskard-scan` may also reach Hugging Face Hub. Do not expect these to run without those secrets and network access.
– `make check` runs `security` (`pip-audit`) and license checks (`check-licenses`/`check-notices`) that reach the network (PyPI / license metadata); if offline these steps can fail even when the code is fine.
