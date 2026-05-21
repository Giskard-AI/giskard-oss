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
