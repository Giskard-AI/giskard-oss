# Rename GreaterEquals → GreaterThanEquals (#2740)

## Plan
- [x] Hard-rename public class `GreaterEquals` to `GreaterThanEquals`
- [x] Keep `@Check.register("greater_than_equals")` unchanged (serialized kind stays loadable)
- [x] Update package exports (`giskard.checks` / `giskard.checks.builtin`)
- [x] Update tests + README
- [x] No deprecated alias (kind string never used the misspelled class name; beta prefers hard delete)

## Review
Hard rename only. Unlike `LesserThan*` (different kind strings needing aliases), `greater_than_equals` was already the registered kind, so deserialization does not require a `GreaterEquals` alias.
