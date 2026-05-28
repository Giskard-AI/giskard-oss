# AUTONOMOUS.md

giskard-oss instructions for autonomous coding agents with no human in the loop.

Read [AGENTS.md](AGENTS.md) first. This file only adds rules that are specific to autonomous execution and PRs; do not duplicate or replace the general workflow, verification, and quality rules from AGENTS.md.

## Setup

Run once before making changes:

```bash
make setup-for-agents AGENT_NAME="<name>" REASON="<issue or task>"
```

Use `make help` for the self-documented command list. Prefer Makefile targets over raw Python or pytest commands.

## Planning

Write the approach to `tasks/todo.md` before touching implementation files. Keep it current as work progresses, and add a short review/results section before opening a PR.

## Stop Conditions

Do not open a PR when the issue is ambiguous, contradictory, or missing acceptance criteria. Post one issue comment with the specific questions needed to proceed, then stop.

Do not silently implement an alternative when you think there is a better approach than requested. Comment with the suggestion and trade-offs, then wait for confirmation.

When responding to PR review, comment back instead of applying the change blindly if the request is unclear, you disagree, or you see a better path.

If no response comes after your single clarifying comment, remain stopped.

## PR Rules

End autonomous-agent PR titles with `🤖🤖🤖🤖`. This marker is required for the expedited-agent PR workflow.

Follow the repository pull request template for the remaining checklist requirements.

## Learning Loop

After any mistake or correction, update AGENTS.md only when the lesson cannot be encoded in scripts, templates, or tests.
