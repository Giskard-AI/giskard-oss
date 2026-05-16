# CLAUDE.md

giskard-oss — behavioral config for Claude Code (interactive assistant with a human in the loop).

## Workflow Orchestration

### 1. Plan Mode Default
– Enter plan mode for ANY non-trivial task (3+ steps or touches multiple libs)
– Write the approach to tasks/todo.md before touching files
– If something goes sideways, STOP and re-plan immediately

### 2. Subagent Strategy
– Use subagents to keep the main context window clean
– One focused task per subagent
– For complex cross-lib problems, spawn parallel subagents

### 3. Self-Improvement Loop
– After ANY correction: update lessons.md
– Write a rule that prevents the same mistake
– Propose additions to CLAUDE.md in the format "Add rule: X"

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
1. Plan First — write plan to tasks/todo.md
2. Verify Plan — check in before starting
3. Track Progress — mark items complete as you go
4. Explain Changes — high-level summary at each step
5. Document Results — add review section to tasks/todo.md
6. Capture Lessons — update lessons.md after corrections

## Core Principles
– Simplicity First: make every change as simple as possible; prefer deleting lines over adding them
– No Laziness: find root causes; no band-aids, no temporary fixes; senior developer standards
– Minimal Impact: only touch what's necessary; no side effects; no reformatting untouched lines
