from typing import Optional, Dict, Any
import json

from giskard.checks.core.check import Check
from giskard.checks.core.result import CheckResult
from giskard.checks.core import Trace, resolve

from pydantic import field_validator


@Check.register("json_valid")
class JsonValid(Check):
    key: Optional[str] = None
    expected_schema: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None  # ✅ FAST FIX (compatibility)

    @field_validator("expected_schema", "schema")
    @classmethod
    def validate_schema(cls, v):
        if v is None:
            return v
        try:
            import jsonschema
            jsonschema.Draft7Validator.check_schema(v)
        except ImportError:
            raise ImportError("The 'jsonschema' library is required.")
        except Exception as e:
            raise ValueError(f"Invalid JSON schema: {str(e)}")
        return v

    async def run(self, trace: Trace) -> CheckResult:
        try:
            # Handle key-based extraction
            if self.key:
                try:
                    value = resolve(trace, self.key)
                except Exception:
                    return CheckResult.failure(message="Key not found in trace")
            else:
                # Support DummyTrace + real Trace
                if hasattr(trace, "last") and trace.last is not None:
                    value = trace.last.output
                elif hasattr(trace, "outputs"):
                    value = trace.outputs
                else:
                    value = trace

            if value is None:
                return CheckResult.failure(message="Output is empty")

            parsed = json.loads(value) if isinstance(value, str) else value

            #  FIX: support both expected_schema and schema
            schema = self.expected_schema or self.schema

            if schema:
                import jsonschema
                try:
                    jsonschema.validate(instance=parsed, schema=schema)
                except Exception as e:
                    return CheckResult.failure(
                        message=f"Schema validation failed: {str(e)}"
                    )

            return CheckResult.success(message="Valid JSON")

        except json.JSONDecodeError as e:
            return CheckResult.failure(message=f"Invalid JSON: {str(e)}")

        except Exception as e:
            return CheckResult.error(message=f"Unexpected error: {str(e)}")