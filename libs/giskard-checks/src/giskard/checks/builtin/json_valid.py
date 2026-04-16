from typing import Optional, Dict, Any
import json

from giskard.checks.core.check import Check
from giskard.checks.core.result import CheckResult
from giskard.core.core import Trace
from giskard.checks.core import resolve
from giskard.checks.core.json_path import JSONPathStr
from giskard.checks.core.exceptions import NoMatch

from pydantic import field_validator


@Check.register("json_valid")
class JsonValid(Check):
    """
    Check if output is valid JSON, optionally validating against a JSON schema.

    Parameters
    ----------
    key : Optional[str]
        JSONPath key to extract value from trace. If None, uses full output.
    expected_schema : Optional[Dict[str, Any]]
        JSON schema to validate against.
    """

    key: Optional[str] = None
    expected_schema: Optional[Dict[str, Any]] = None

    #  Schema validation using Pydantic
    @field_validator("expected_schema")
    @classmethod
    def validate_schema(cls, v):
        if v is None:
            return v
        try:
            import jsonschema
            jsonschema.Draft7Validator.check_schema(v)
        except ImportError:
            raise ImportError(
                "The 'jsonschema' library is required for schema validation."
            )
        except Exception as e:
            raise ValueError(f"Invalid JSON schema: {str(e)}")
        return v

    async def run(self, trace: Trace) -> CheckResult:
        try:
            try:
                #  ALWAYS use resolve (no manual trace access)
                if self.key:
                    value = resolve(trace, JSONPathStr(self.key))
                else:
                    value = resolve(trace, JSONPathStr("$"))
            except NoMatch:
                return CheckResult.failure(message="Key not found in trace")

            if value is None:
                return CheckResult.failure(message="Output is empty")

            parsed = json.loads(value) if isinstance(value, str) else value

            if self.expected_schema:
                import jsonschema
                try:
                    jsonschema.validate(instance=parsed, schema=self.expected_schema)
                except Exception as e:
                    return CheckResult.failure(
                        message=f"Schema validation failed: {str(e)}"
                    )

            return CheckResult.success(message="Valid JSON")

        except json.JSONDecodeError as e:
            return CheckResult.failure(message=f"Invalid JSON: {str(e)}")

        except Exception as e:
            return CheckResult.error(message=f"Unexpected error: {str(e)}")