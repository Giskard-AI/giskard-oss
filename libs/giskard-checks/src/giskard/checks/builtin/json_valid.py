"""JSON validation check implementation."""

import json
from typing import Any, override

from jsonschema import SchemaError, validate
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, field_validator
from referencing.exceptions import Unresolvable

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult


@Check.register("json_valid")
class JsonValid[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Check that validates whether a trace value is valid JSON.

    With ``parse=True`` (default), the extracted value must be a serialized
    JSON string, which is parsed with ``json.loads`` before validation. With
    ``parse=False``, the value is treated as an already-parsed JSON value
    (dict, list, str, number, bool, or None) and only checked for JSON
    serializability and schema conformance.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the value to validate.",
    )
    parse: bool = Field(
        default=True,
        description=(
            "If True, the value is treated as a serialized JSON string and "
            "parsed before validation. If False, the value is treated as an "
            "already-parsed JSON value."
        ),
    )
    expected_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="Optional JSON Schema to validate the parsed JSON value against.",
    )

    @field_validator("expected_schema")
    @classmethod
    def validate_schema_definition(
        cls, schema: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if schema is None:
            return schema

        try:
            cls._validate_schema_definition(schema)
        except SchemaError as err:
            raise ValueError(
                f"Provided JSON Schema is invalid: {err.message}."
            ) from err
        except Unresolvable as err:
            raise ValueError(
                f"Provided JSON Schema contains an unresolvable reference: {err}."
            ) from err

        return schema

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        value = resolve(trace, self.key)
        details: dict[str, Any] = {
            "key": self.key,
            "value": value,
            "schema": self.expected_schema,
        }

        if isinstance(value, NoMatch):
            return CheckResult.failure(
                message=f"No value found for key '{self.key}'.",
                details=details,
            )

        if self.parse:
            if not isinstance(value, str):
                return CheckResult.failure(
                    message=f"Value at key '{self.key}' is not a string: {value!r}",
                    details=details,
                )
            try:
                value = json.loads(value)
            except json.JSONDecodeError as err:
                details["error"] = str(err)
                return CheckResult.failure(
                    message=f"Value at key '{self.key}' is not valid JSON: {err}",
                    details=details,
                )
        else:
            try:
                json.dumps(value)
            except (TypeError, ValueError) as err:
                details["error"] = str(err)
                return CheckResult.failure(
                    message=f"Value at key '{self.key}' is not JSON serializable: {err}",
                    details=details,
                )

        details["parsed_value"] = value

        if self.expected_schema is not None:
            try:
                self._validate_schema(value, self.expected_schema)
            except Unresolvable as err:
                details["error"] = str(err)
                return CheckResult.error(
                    message=f"JSON Schema contains an unresolvable $ref: {err}.",
                    details=details,
                )
            except JsonSchemaValidationError as err:
                details["error"] = err.message
                return CheckResult.failure(
                    message=(
                        f"JSON value at key '{self.key}' does not match the "
                        f"provided schema: {err.message}."
                    ),
                    details=details,
                )

        return CheckResult.success(
            message=f"Value at key '{self.key}' is valid JSON.",
            details=details,
        )

    @staticmethod
    def _validate_schema_definition(schema: dict[str, Any]) -> None:
        validator_for(schema).check_schema(schema)

    @staticmethod
    def _validate_schema(parsed_value: Any, schema: dict[str, Any]) -> None:
        validate(instance=parsed_value, schema=schema)
