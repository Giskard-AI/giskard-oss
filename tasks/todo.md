# Issue #2731 — root giskard-scan extra pins

## Approach

1. Bump root `scan` / `garak` / `deepteam` lower bounds from `1.0.0b2` → `1.0.0b4`.
2. Add `tools/check_extra_pins.py` + `make check-extra-pins` into `make check`.
3. Verify pass + intentional-drift fail; run `make format && make check`.

## Progress

- [x] Plan at `docs/plans/2026-08-12-001-fix-root-scan-extra-pins-plan.md`
- [x] Pin bump in root `pyproject.toml`
- [x] Check script + Makefile wiring
- [x] Drift smoke (pass / fail / restore)
- [ ] Full `make check` green
- [ ] PR title ends with 🤖🤖🤖🤖

## Review / results

- Drift smoke confirmed: stale `b2` on `scan` fails with expected/found message.
- basedpyright initially flagged untyped `dict` on the helper; fixed with `dict[str, Any]` and removed `from __future__ import annotations` per repo Python rules.
