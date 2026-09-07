---
name: fix-cve
description: Use when pip-audit (or a CVE/GHSA advisory, Dependabot, or security scan) flags a vulnerable Python dependency in giskard-oss and it needs upgrading to a fixed version in the uv lockfile.
---

# fix-cve

Remediate a vulnerable dependency flagged by `pip-audit` (or any CVE/GHSA advisory) by upgrading it in the **uv lockfile**, then verify with the repo gate.

`pip-audit` runs as part of `make check` (the `security` target). The advisory looks like:

```
Name    Version ID                  Fix Versions
------- ------- ------------------- ------------
msgpack 1.1.2   GHSA-6v7p-g79w-8964 1.2.1
```

You need: the **package name** and a **fix version**.

## Procedure

```bash
# 1. Upgrade the package in the lockfile (works for direct AND transitive deps).
uv lock --upgrade-package <package>

# 2. Apply to the environment.
uv sync

# 3. Verify the advisory is gone — re-run the exact audit command.
uv run pip-audit --skip-editable
```

Then run the full repo gate (it re-runs `pip-audit` via the `security` target, plus format/lint/types/tests):

```bash
make format && make check && make test-unit
```

Or invoke `/check` (the project verification-gate skill).

## Key facts

- **Always upgrade in the lockfile, not the environment.** Use `uv lock --upgrade-package <package>`, NOT `pip install` / `uv pip install`. The lockfile is the source of truth; a `pip install` fix is lost on the next `uv sync`.
- **Transitive deps are fixed the same way.** Most flagged packages (e.g. `msgpack`, pulled in by `cachecontrol`) are not in any `pyproject.toml`. You still upgrade them by name with `uv lock --upgrade-package <package>` — no `pyproject.toml` edit needed. Do NOT add a direct dependency just to pin it.
- **`uv lock --upgrade-package <package>` takes the newest version the constraints allow.** Check the resulting `uv.lock` entry to confirm the version actually moved to the fix version — a cap may have held it below.

### When a constraint caps the version below the fix

If another dependency caps the package below the fix version, the bare upgrade silently stays on the old version, and forcing it with `uv lock --upgrade-package "<package>>=<fix-version>"` **fails with a resolution error** (`No solution found...`) — uv constraints can only *narrow* the version set, never expand past a cap. Escalate in this order:

1. **Upgrade the capping parent.** Find who imposes the cap (`uv tree --invert --package <package>`), then `uv lock --upgrade-package <parent>`. A newer parent often relaxes the cap. Preferred — keeps the dependency graph honest.
2. **Override the transitive dep** if no parent upgrade resolves it. Add to `pyproject.toml`:
   ```toml
   [tool.uv]
   override-dependencies = ["<package>>=<fix-version>"]
   ```
   Overrides *expand* the allowed set (the only escape hatch a cap can't block), then re-run `uv lock`. Use sparingly — it ignores the parent's declared constraint, so re-verify nothing breaks with the full gate.
- **Lockfile-only changes have no "affected lib".** Run the full `make test-unit` (all packages), not a single `PACKAGE=<lib>` scope — the dep could affect any lib.

## Verifying the fix

Do not assert "fixed" — show evidence:

1. `uv run pip-audit --skip-editable` prints **"No known vulnerabilities found"** (editable `giskard-*` libs in the skip list are expected and harmless).
2. The `uv.lock` diff shows the package moved to the fix version (`Updated <pkg> vX -> vY` from `uv lock`).
3. `make check` passes (its `security` target re-runs the audit).

## Common mistakes

| Mistake | Fix |
|---------|-----|
| `pip install <pkg>==<ver>` to fix it | Use `uv lock --upgrade-package <pkg>`; pip changes don't persist in the lockfile |
| Editing `pyproject.toml` for a transitive dep | Not needed — `uv lock --upgrade-package` handles transitive deps by name |
| Skipping re-audit, assuming the upgrade worked | Re-run `uv run pip-audit --skip-editable`; a constraint may have silently capped the version below the fix |
| Force-pinning past a cap with `"<pkg>>=<fix>"` and hitting `No solution found` | Constraints can't expand past a cap — upgrade the capping parent, or use `[tool.uv] override-dependencies` |
| Guessing the verification command | Run `/check` or `make format && make check && make test-unit`; `make check` includes `pip-audit` |
| Scoping tests to one `PACKAGE=` | A lockfile change has no single affected lib; run all unit tests |
