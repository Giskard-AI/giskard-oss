import json
from typing import Any, Optional, Dict

from giskard.checks.core import Check, CheckResult, CheckStatus, Trace


@Check.register("json_valid")
class JsonValid(Check):
    key: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

   async def run(self, trace: Trace) -> CheckResult:
    try:
        # Handle trace safely
        if hasattr(trace, "last"):
            if trace.last is None:
                return CheckResult.failure(message="Trace is empty")
            value = trace.last.outputs
        else:
            value = trace.outputs

        if value is None:
            return CheckResult.failure(message="Output is empty")

        # Parse JSON only if string
        parsed = json.loads(value) if isinstance(value, (str, bytes)) else value

        # Schema validation
        if self.schema:
            try:
                import jsonschema
            except ImportError:
                return CheckResult.error(
                    message="The 'jsonschema' library is required for schema validation."
                )

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
        return CheckResult.error(
            message=f"Unexpected error: {str(e)}"
        )