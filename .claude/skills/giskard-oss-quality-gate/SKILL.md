---
name: giskard-oss-quality-gate
description: Run the full code quality gate for the giskard-oss monorepo—format, lint, typecheck, and unit tests across all packages. Use when finishing a feature, before opening a PR, or when the user asks to check quality or run the gate.
disable-model-invocation: true
allowed-tools: Bash(make *)
---

# giskard-oss — quality gate

## Results

```!
echo "=== make format ===" && make format 2>&1 | tail -5
echo "=== make check ===" && make check 2>&1 | tail -10
echo "=== make test-unit PACKAGE=giskard-core ===" && make test-unit PACKAGE=giskard-core 2>&1 | tail -5
echo "=== make test-unit PACKAGE=giskard-checks ===" && make test-unit PACKAGE=giskard-checks 2>&1 | tail -5
echo "=== make test-unit PACKAGE=giskard-agents ===" && make test-unit PACKAGE=giskard-agents 2>&1 | tail -5
```

Report pass/fail for each step. Fix root causes on failure — no `# type: ignore`, no skips. Re-run after fixing.
