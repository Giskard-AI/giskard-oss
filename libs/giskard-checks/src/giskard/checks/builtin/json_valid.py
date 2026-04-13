from typing import Optional, Dict, Any
import json

from giskard.checks.core.check import Check
from giskard.checks.core.result import CheckResult
from giskard.checks.core import Trace


@Check.register("json_valid")
class JsonValid(Check):
    """
    Check if output is valid JSON, optionally validating against a JSON schema.
    """

    key: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        key: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(key=key, schema=schema)

        if schema is not None:
            try:
                import jsonschema
                jsonschema.Draft7Validator.check_schema(schema)
            except ImportError:
                raise ImportError(
                    "The 'jsonschema' library is required for schema validation."
                )
            except Exception as e:
                raise ValueError(f"Invalid JSON schema: {str(e)}")

    async def run(self, trace: Trace) -> CheckResult:
        try:
            # Safe extraction (handles DummyTrace + real Trace)
            if self.key:
                if hasattr(trace, "last") and trace.last is not None:
                    value = trace.last.output.get(self.key)
                else:
                    value = trace.outputs.get(self.key)
            else:
                if hasattr(trace, "last") and trace.last is not None:
                    value = trace.last.output
                else:
                    value = trace.outputs

            if value is None:
                return CheckResult.failure(message="Output is empty")

            parsed = json.loads(value) if isinstance(value, str) else value

            if self.schema:
                import jsonschema
                try:
                    jsonschema.validate(instance=parsed, schema=self.schema)
                except Exception as e:
                    return CheckResult.failure(
                        message=f"Schema validation failed: {str(e)}"
                    )

            return CheckResult.success(message="Valid JSON")

        except json.JSONDecodeError as e:
            return CheckResult.failure(message=f"Invalid JSON: {str(e)}")

        except Exception as e:
            return CheckResult.error(message=f"Unexpected error: {str(e)}")