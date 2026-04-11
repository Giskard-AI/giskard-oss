import json
from typing import Any, Optional, Dict

from giskard.checks.core import Check, CheckResult, CheckStatus, Trace


@Check.register("json_valid")
class JsonValid(Check):
    key: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None

    async def run(self, trace: Trace) -> CheckResult:
        try:
            # ✅ Support both real Trace and DummyTrace
            if hasattr(trace, "last"):
                value = trace.last.outputs
            else:
                value = trace.outputs

            # Parse JSON
            parsed = json.loads(value)

            # Schema validation
            if self.schema:
                try:
                    import jsonschema
                    jsonschema.validate(instance=parsed, schema=self.schema)
                except Exception as e:
                    return CheckResult(
                        status=CheckStatus.FAIL,
                        message=f"Schema validation failed: {str(e)}",
                    )

            return CheckResult(
                status=CheckStatus.PASS,
                message="Valid JSON",
            )

        except Exception as e:
            return CheckResult(
                status=CheckStatus.FAIL,
                message=f"Invalid JSON: {str(e)}",
            )