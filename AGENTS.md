# AGENTS.md

giskard-oss — behavioral config for autonomous coding agents (no human in the loop). Human-oriented docs: [README.md](README.md). Always invoke Python via `uv run` — bare `python` or `pytest` will fail.

## Workflow Orchestration

### 1. Setup
Run once: `make setup-for-agents AGENT_NAME="<name>" REASON="<issue or task>"`
If tools are missing later: `make install && make install-tools`

### 2. Plan Before Acting
– Write approach to tasks/todo.md before touching any file
– If scope is unclear, read `libs/<pkg>/.cursor/rules/` before proceeding
– If something goes wrong, stop and re-plan

### 3. Verification Before Done
– Before any PR: `make format && make check && make test-unit PACKAGE=<affected-lib>`
– Show `make check` and pytest summary output in the PR description
– No `# type: ignore` without structural-fix explanation; no patching test assertions

### 4. PR Rules
– End PR titles with `🤖🤖🤖🤖` — required for the expedited-agent PR workflow
– Minimal diffs: implement exactly what was asked

### 5. Self-Improvement Loop
– After any mistake: update lessons.md with what went wrong and what rule would have prevented it

### 6. Clarify Before Acting
– When an issue is ambiguous, contradictory, or missing acceptance criteria: **do not open a PR**. Post a comment on the issue asking the specific questions needed to proceed, then stop.
– When you have a better approach than what was requested: **do not implement your alternative silently**. Comment on the issue explaining your suggestion and the trade-offs, then wait for confirmation.
– When responding to a PR review comment: if the requested change is unclear, you disagree, or you see a better path: **comment back with your question or counter-proposal**, do not just apply the change blindly.
– One comment is enough — do not loop. If no response comes, remain stopped.
– When scope is unambiguous and you have no better suggestion: proceed directly without asking.

## Task Management
1. Plan First — write plan to tasks/todo.md
2. Verify Plan — review before starting; proceed if unambiguous
3. Track Progress — mark items complete as you go
4. Explain Changes — high-level summary at each step
5. Document Results — add review section to tasks/todo.md
6. Capture Lessons — update lessons.md after corrections

## Core Principles
– Simplicity First: make every change as simple as possible; prefer deleting lines over adding them
– No Laziness: find root causes; no band-aids, no temporary fixes; senior developer standards
– Minimal Impact: only touch what's necessary; no side effects; no reformatting untouched lines

---
This file follows the open [AGENTS.md](https://agents.md/) convention for coding agents.
