from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from giskard.core.discriminated import _REGISTRY
from giskard.core.utils import NOT_PROVIDED, NotProvided

from . import builtin, judges, testing  # noqa: F401
from .core.check import Check
from .core.interaction import InteractionSpec, Trace
from .core.result import ScenarioResult, SuiteResult
from .core.scenario import Scenario, Step
from .scenarios.suite import Suite

console = Console()
error_console = Console(stderr=True)


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class LoadedDefinition:
    kind: str
    value: Scenario[Any, Any, Any] | Suite[Any, Any]
    path: Path


def _load_file(path: Path) -> Any:
    if not path.exists():
        raise CliError(f"Definition file not found: {path}")

    suffix = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"Unable to read definition file {path}: {exc}") from exc

    try:
        if suffix == ".json":
            return json.loads(raw)
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CliError(f"Failed to parse {path.name}: {exc}") from exc

    raise CliError(
        f"Unsupported file type for {path.name}. Expected .json, .yaml, or .yml."
    )


def _normalize_items(
    singular: str,
    plural: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in (singular, plural):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            candidate_items = value
        else:
            candidate_items = [value]

        for item in candidate_items:
            if not isinstance(item, dict):
                raise CliError(f"Expected '{key}' entries to be objects, got {item!r}.")
            normalized = dict(item)
            if singular == "interact":
                normalized.setdefault("kind", "interact")
            items.append(normalized)
    return items


def _ensure_allowed_keys(
    payload: dict[str, Any], *, allowed: set[str], context: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CliError(
            f"Unexpected keys in {context}: {', '.join(unknown)}. Allowed keys: {', '.join(sorted(allowed))}."
        )


def _module_search_paths(path: Path) -> list[str]:
    search_paths = [str(path.parent.resolve()), str(Path.cwd().resolve())]
    unique_paths: list[str] = []
    for candidate in search_paths:
        if candidate not in unique_paths:
            unique_paths.append(candidate)
    return unique_paths


def _module_belongs_to_search_path(module: Any, search_path: str) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False

    module_path = Path(module_file).resolve()
    return module_path.is_relative_to(Path(search_path))


def _module_belongs_to_search_paths(module: Any, search_paths: list[str]) -> bool:
    return any(
        _module_belongs_to_search_path(module, search_path)
        for search_path in search_paths
    )


def _module_exists_in_search_path(module_name: str, search_path: str) -> bool:
    module_parts = module_name.split(".")
    module_root = Path(search_path).resolve().joinpath(*module_parts)
    return module_root.with_suffix(".py").exists() or (
        module_root / "__init__.py"
    ).exists()


def _resolve_python_target(value: Any, *, path: Path) -> Any:
    if value is None:
        return NOT_PROVIDED
    if isinstance(value, NotProvided):
        return value
    if not isinstance(value, str):
        raise CliError(
            f"Invalid target in {path.name}: expected a string like 'python:module.symbol'."
        )
    if not value.startswith("python:"):
        raise CliError(
            f"Unsupported target '{value}' in {path.name}. Only 'python:module.symbol' is supported."
        )

    import_path = value.removeprefix("python:")
    module_name, _, attribute_path = import_path.rpartition(".")
    if not module_name or not attribute_path:
        raise CliError(
            f"Invalid python target '{value}' in {path.name}. Expected 'python:module.symbol'."
        )

    search_paths = _module_search_paths(path)
    original_sys_path = sys.path[:]
    try:
        for search_path in reversed(search_paths):
            if search_path not in sys.path:
                sys.path.insert(0, search_path)

        cached_module = sys.modules.get(module_name)
        if cached_module is not None:
            preferred_search_path = search_paths[0]
            if _module_exists_in_search_path(module_name, preferred_search_path):
                if not _module_belongs_to_search_path(
                    cached_module, preferred_search_path
                ):
                    sys.modules.pop(module_name, None)
            elif not _module_belongs_to_search_paths(cached_module, search_paths):
                sys.modules.pop(module_name, None)

        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise CliError(
            f"Could not import module '{module_name}' for target '{value}' in {path.name}: {exc}"
        ) from exc
    finally:
        sys.path[:] = original_sys_path

    target: Any = module
    try:
        for attribute in attribute_path.split("."):
            target = getattr(target, attribute)
    except AttributeError as exc:
        raise CliError(
            f"Could not resolve attribute '{attribute_path}' for target '{value}' in {path.name}."
        ) from exc

    if not callable(target):
        raise CliError(
            f"Resolved target '{value}' in {path.name}, but it is not callable."
        )

    return target


def _build_step(data: Any) -> Step[Any, Any, Any]:
    if not isinstance(data, dict):
        raise CliError(f"Each step must be an object, got {data!r}.")

    _ensure_allowed_keys(
        data,
        allowed={"interact", "interacts", "check", "checks"},
        context="a scenario step",
    )

    interactions = [
        InteractionSpec[Any, Any, Any].model_validate(item)
        for item in _normalize_items("interact", "interacts", data)
    ]
    checks = [
        Check[Any, Any, Any].model_validate(item)
        for item in _normalize_items("check", "checks", data)
    ]
    return Step[Any, Any, Any](interacts=interactions, checks=checks)


def _build_scenario(data: Any, *, path: Path) -> Scenario[Any, Any, Any]:
    if not isinstance(data, dict):
        raise CliError(f"Scenario definition in {path.name} must be an object.")

    _ensure_allowed_keys(
        data,
        allowed={"annotations", "name", "steps", "target", "trace_type"},
        context=f"scenario definition in {path.name}",
    )

    steps = data.get("steps")
    if not isinstance(steps, list):
        raise CliError(
            f"Scenario definition in {path.name} must contain a 'steps' list."
        )

    scenario_data = {
        "name": data.get("name", path.stem),
        "steps": [_build_step(step) for step in steps],
        "annotations": data.get("annotations", {}),
        "target": _resolve_python_target(data.get("target"), path=path),
        "trace_type": data.get("trace_type"),
    }

    trace_type = scenario_data["trace_type"]
    if isinstance(trace_type, str):
        trace_type = _resolve_python_target(trace_type, path=path)
        if not isinstance(trace_type, type) or not issubclass(trace_type, Trace):
            raise CliError(
                f"Resolved trace_type '{scenario_data['trace_type']}' in {path.name}, but it is not a Trace subclass."
            )
        scenario_data["trace_type"] = trace_type

    return Scenario[Any, Any, Any].model_validate(scenario_data)


def _build_suite(data: Any, *, path: Path) -> Suite[Any, Any]:
    if not isinstance(data, dict):
        raise CliError(f"Suite definition in {path.name} must be an object.")

    _ensure_allowed_keys(
        data,
        allowed={"name", "scenarios", "target"},
        context=f"suite definition in {path.name}",
    )

    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise CliError(
            f"Suite definition in {path.name} must contain a 'scenarios' list."
        )

    suite_data = {
        "name": data.get("name", path.stem),
        "scenarios": [_build_scenario(scenario, path=path) for scenario in scenarios],
        "target": _resolve_python_target(data.get("target"), path=path),
    }
    return Suite[Any, Any].model_validate(suite_data)


def load_definition(path_str: str) -> LoadedDefinition:
    path = Path(path_str).expanduser().resolve()
    payload = _load_file(path)

    if not isinstance(payload, dict):
        raise CliError(f"Definition file {path.name} must contain a top-level object.")

    try:
        if "scenarios" in payload:
            return LoadedDefinition("suite", _build_suite(payload, path=path), path)
        if "steps" in payload:
            return LoadedDefinition("scenario", _build_scenario(payload, path=path), path)
    except ValidationError as exc:
        raise CliError(f"Invalid definition in {path.name}:\n{exc}") from exc

    raise CliError(
        f"Unable to determine definition type for {path.name}. Expected top-level 'steps' or 'scenarios'."
    )


def _build_suite_result(
    result: ScenarioResult[Any] | SuiteResult,
) -> SuiteResult:
    if isinstance(result, SuiteResult):
        return result
    return SuiteResult(results=[result], duration_ms=result.duration_ms)


def _write_text_output(output: str, path: Path | None) -> None:
    if path is None:
        print(output)
        return
    path.write_text(output, encoding="utf-8")


def _result_exit_code(result: ScenarioResult[Any] | SuiteResult) -> int:
    suite_result = _build_suite_result(result)
    return 0 if not suite_result.failures_and_errors else 1


def _render_json(result: ScenarioResult[Any] | SuiteResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, default=str)


def _run_command(args: argparse.Namespace) -> int:
    loaded = load_definition(args.path)
    execution = asyncio.run(
        cast(Any, loaded.value).run(return_exception=True)  # pyright: ignore[reportAttributeAccessIssue]
    )

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if args.format == "rich":
        if output_path is not None:
            raise CliError("The rich format does not support --output. Use json or junit.")
        console.print(execution)
    elif args.format == "json":
        _write_text_output(_render_json(execution), output_path)
    elif args.format == "junit":
        xml = _build_suite_result(execution).to_junit_xml(path=output_path)
        if output_path is None:
            print(xml)
    else:
        raise CliError(f"Unsupported output format: {args.format}")

    return _result_exit_code(execution)


def _validate_summary(loaded: LoadedDefinition) -> dict[str, Any]:
    if loaded.kind == "scenario":
        scenario = cast(Scenario[Any, Any, Any], loaded.value)
        return {
            "valid": True,
            "type": "scenario",
            "name": scenario.name,
            "steps": len(scenario.steps),
        }

    suite = cast(Suite[Any, Any], loaded.value)
    return {
        "valid": True,
        "type": "suite",
        "name": suite.name,
        "scenarios": len(suite.scenarios),
    }


def _validate_command(args: argparse.Namespace) -> int:
    loaded = load_definition(args.path)
    summary = _validate_summary(loaded)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        if loaded.kind == "scenario":
            console.print(
                f"[green]Valid scenario[/green] '{summary['name']}' with {summary['steps']} step(s)."
            )
        else:
            console.print(
                f"[green]Valid suite[/green] '{summary['name']}' with {summary['scenarios']} scenario(s)."
            )

    return 0


def _iter_registered_checks() -> list[tuple[str, type[Any]]]:
    registered = _REGISTRY._reverse_kinds.get(Check, {})
    checks = [
        (kind, cls)
        for kind, cls in registered.items()
        if cls.__module__.startswith("giskard.checks.")
        and ".testing." not in cls.__module__
        and ".tests." not in cls.__module__
    ]
    return sorted(checks, key=lambda item: item[0])


def _list_checks_command(args: argparse.Namespace) -> int:
    registered_checks = _iter_registered_checks()
    if args.format == "json":
        payload = [
            {
                "kind": kind,
                "class_name": cls.__name__,
                "module": cls.__module__,
                "summary": (inspect.getdoc(cls) or "").splitlines()[0]
                if inspect.getdoc(cls)
                else "",
            }
            for kind, cls in registered_checks
        ]
        print(json.dumps(payload, indent=2))
        return 0

    table = Table(title="Available Checks")
    table.add_column("Kind")
    table.add_column("Class")
    table.add_column("Summary")
    for kind, cls in registered_checks:
        doc = inspect.getdoc(cls) or ""
        summary = doc.splitlines()[0] if doc else ""
        table.add_row(kind, cls.__name__, summary)
    console.print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="giskard",
        description="Run and validate Giskard scenarios and suites from YAML or JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a scenario or suite definition from YAML or JSON."
    )
    run_parser.add_argument("path", help="Path to a .yaml, .yml, or .json definition.")
    run_parser.add_argument(
        "--format",
        choices=("rich", "json", "junit"),
        default="rich",
        help="Output format for run results.",
    )
    run_parser.add_argument(
        "--output",
        help="Optional file path for json or junit output.",
    )
    run_parser.set_defaults(handler=_run_command)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a scenario or suite definition without running it."
    )
    validate_parser.add_argument(
        "path", help="Path to a .yaml, .yml, or .json definition."
    )
    validate_parser.add_argument(
        "--format",
        choices=("rich", "json"),
        default="rich",
        help="Output format for validation results.",
    )
    validate_parser.set_defaults(handler=_validate_command)

    list_parser = subparsers.add_parser(
        "list", help="List available objects exposed by the CLI."
    )
    list_subparsers = list_parser.add_subparsers(dest="list_command", required=True)
    checks_parser = list_subparsers.add_parser(
        "checks", help="List built-in checks that can be referenced in definitions."
    )
    checks_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format for the checks listing.",
    )
    checks_parser.set_defaults(handler=_list_checks_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        handler = args.handler
        return handler(args)
    except CliError as exc:
        error_console.print(f"[red]{exc}[/red]")
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
