# Credentials & Secrets

Overview of secrets, environments, and authentication used by GitHub Actions workflows.

## Environments

| Environment | Purpose | Used by |
|---|---|---|
| `CI` | Integration/functional tests | `integration-tests.yml` |
| `release` | Package releases to PyPI | `release.yml` |
| `github-pages` | GitHub Pages deployment | (unused) |
| `main` | — | (unused) |
| `test` | — | (unused) |

## Secrets

### Environment: CI

| Secret | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `integration-tests.yml` | Google Gemini API key for functional tests |

### Environment: release

| Secret | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `release.yml` | Google Gemini API key for pre-release tests |

### PyPI publishing

PyPI trusted publishing (OIDC) is used — no static PyPI credentials stored. Each package must have a trusted publisher configured on PyPI:

| Package | PyPI settings URL |
|---|---|
| `giskard-core` | https://pypi.org/manage/project/giskard-core/settings/publishing/ |
| `giskard-agents` | https://pypi.org/manage/project/giskard-agents/settings/publishing/ |
| `giskard-checks` | https://pypi.org/manage/project/giskard-checks/settings/publishing/ |

Publisher config: Owner `Giskard-AI`, Repository `giskard-oss`, Workflow `release.yml`, Environment `release`.

## Pending cleanup

| Item | Action |
|---|---|
| `PIPY_USERNAME` / `PIPY_PASSWORD` | Delete after confirming trusted publishing works for all packages |
| `RELEASE_PAT_TOKEN` (org-level) | Delete — no longer referenced, replaced by `GITHUB_TOKEN` |
| `github-pages`, `main`, `test` environments | Review — currently unused, candidates for removal |

## Dependency update cooldown

Both Dependabot and Renovate are configured with a 7-day cooldown before proposing dependency updates (`dependabot.yml` → `default-days: 7`, `renovate.json` → `minimumReleaseAge: "7 days"`).
