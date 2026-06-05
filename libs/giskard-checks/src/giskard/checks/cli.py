import argparse
import asyncio
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.table import Table

from . import builtin, judges, testing  # noqa: F401
from .core.check import Check
from .core.importing import python_reference_path
from .core.result import ScenarioResult, SuiteResult
from .core.scenario import Scenario
from .scenarios.suite import Suite

console = Console()
error_console = Console(stderr=True)


class CliError(Exception):
    """Raised for expected CLI errors with a specific exit code.

    Parameters
    ----------
    message : str
        Human-readable error description.
    exit_code : int, optional
        Process exit code to use, by default 2.
    """

    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class LoadedDefinition:
    """A parsed and validated scenario or suite definition.

    Attributes
    ----------
    kind : str
        Either ``"scenario"`` or ``"suite"``.
    value : Scenario | Suite
        The validated definition object.
    path : Path
        Absolute path to the source file.
    """

    kind: str
    value: Scenario[Any, Any, Any] | Suite[Any, Any]
    path: Path


def _load_yaml(raw: str, path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise CliError(
            "YAML support requires PyYAML. Install `giskard-checks[yaml]` or use a JSON definition file."
        ) from exc

    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CliError(f"Failed to parse {path.name}: {exc}") from exc


def _load_file(path: Path) -> Any:
    if not path.exists():
        raise CliError(f"Definition file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"Unable to read definition file {path}: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliError(f"Failed to parse {path.name}: {exc}") from exc

    if suffix in {".yaml", ".yml"}:
        return _load_yaml(raw, path)

    raise CliError(
        f"Unsupported file type for {path.name}. Expected .json, .yaml, or .yml."
    )


def load_definition(path_str: str) -> LoadedDefinition:
    """Load and validate a scenario or suite definition file.

    Parameters
    ----------
    path_str : str
        Path to a ``.json``, ``.yaml``, or ``.yml`` definition file.

    Returns
    -------
    LoadedDefinition
        The kind, validated value, and resolved absolute path.

    Raises
    ------
    CliError
        If the file cannot be found, read, parsed, or validated.
    """
    path = Path(path_str).expanduser().resolve()
    payload = _load_file(path)

    if not isinstance(payload, dict):
        raise CliError(f"Definition file {path.name} must contain a top-level object.")

    if "scenarios" not in payload and "steps" not in payload:
        raise CliError(
            f"Unable to determine definition type for {path.name}. Expected top-level 'steps' or 'scenarios'."
        )

    definition_kind = "suite" if "scenarios" in payload else "scenario"
    adapter: TypeAdapter[Suite[Any, Any] | Scenario[Any, Any, Any]]
    if definition_kind == "suite":
        adapter = TypeAdapter(Suite[Any, Any])
    else:
        adapter = TypeAdapter(Scenario[Any, Any, Any])

    try:
        with python_reference_path(path):
            definition = adapter.validate_python(payload, context={"path": path})
    except ValidationError as exc:
        raise CliError(f"Invalid definition in {path.name}:\n{exc}") from exc

    if definition_kind == "scenario":
        return LoadedDefinition("scenario", definition, path)

    return LoadedDefinition("suite", definition, path)


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

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    except OSError as exc:
        raise CliError(f"Failed to write output to {path}: {exc}") from exc


def _result_exit_code(result: ScenarioResult[Any] | SuiteResult) -> int:
    suite_result = _build_suite_result(result)
    return 0 if not suite_result.failures_and_errors else 1


def _render_json(result: ScenarioResult[Any] | SuiteResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, default=str)


def _run_command(args: argparse.Namespace) -> int:
    """Execute the ``giskard run`` sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing ``path``, ``format``, and ``output``.

    Returns
    -------
    int
        0 if all checks passed, 1 if any check failed or errored.
    """
    loaded = load_definition(args.path)
    execution = asyncio.run(
        cast(Any, loaded.value).run(return_exception=True)  # pyright: ignore[reportAttributeAccessIssue]
    )

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if args.format == "rich":
        if output_path is not None:
            raise CliError(
                "The rich format does not support --output. Use json or junit."
            )
        console.print(execution)
    elif args.format == "json":
        _write_text_output(_render_json(execution), output_path)
    elif args.format == "junit":
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
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
    """Execute the ``giskard validate`` sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing ``path`` and ``format``.

    Returns
    -------
    int
        0 on success, 2 if the definition is invalid.
    """
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
    registered = Check.list_registered_types()
    checks = [
        (kind, cls)
        for kind, cls in registered.items()
        if cls.__module__.startswith("giskard.checks.")
        and ".testing." not in cls.__module__
        and ".tests." not in cls.__module__
    ]
    return sorted(checks, key=lambda item: item[0])


def _list_checks_command(args: argparse.Namespace) -> int:
    """Execute the ``giskard list checks`` sub-command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments containing ``format``.

    Returns
    -------
    int
        Always 0.
    """
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
    """Build the top-level argument parser for the ``giskard`` CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with ``run``, ``validate``, and ``list`` sub-commands.
    """
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
    """Entry point for the ``giskard`` CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on check failures, 2 on usage errors.
    """
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
