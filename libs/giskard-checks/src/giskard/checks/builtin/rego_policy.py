"""Built-in check that evaluates Rego policies against trace data via regorus."""

import importlib
import json
import warnings
from typing import Any, Self, override

from pydantic import Field, PrivateAttr, model_validator

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, resolve
from ..core.result import CheckResult

_REGORUS_INSTALL_HINT = (
    "The regorus dependency is required to run RegoPolicy. "
    "Install it with: pip install 'giskard-checks[regorus]' "
    "(installs the celine-regorus PyPI wheel; imports as regorus)."
)

_POLICY_FILENAME = "giskard_policy.rego"
_VALIDATION_SKIPPED_WARNING = (
    "RegoPolicy validation skipped: optional dependency 'regorus' is not installed. "
    "Install it with: pip install 'giskard-checks[regorus]'."
)


def _result_from_boolean_rule(
    value: Any,
    *,
    rule: str,
    details: dict[str, Any],
) -> CheckResult:
    details = {**details, "value": value, "rule": rule}

    if value is True:
        return CheckResult.success(
            message=f"Rule '{rule}' evaluated to true.",
            details=details,
        )
    if value is False:
        return CheckResult.failure(
            message=f"Rule '{rule}' evaluated to false.",
            details=details,
        )
    if value is None:
        return CheckResult.failure(
            message=f"Rule '{rule}' is undefined.",
            details=details,
        )
    return CheckResult.error(
        message=(
            f"Rule '{rule}' returned {type(value).__name__}, "
            "expected bool or undefined."
        ),
        details=details,
    )


@Check.register("rego_policy")
class RegoPolicy[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Check[InputType, OutputType, TraceType]
):
    """Evaluate an inline Rego policy against trace data using regorus.

    Extracts the OPA ``input`` document from the trace via JSONPath, merges
    optional static ``data``, and evaluates a boolean ``rule``. Requires the
    optional ``regorus`` extra (``pip install 'giskard-checks[regorus]'``,
    installs ``celine-regorus`` from PyPI; ``import regorus`` at runtime).

    The evaluated rule must return a boolean, be undefined (fail), or raise an
    error if it returns another type.

    Attributes
    ----------
    policy : str
        Inline Rego source loaded into the engine.
    rule : str
        Fully qualified boolean rule path (e.g. ``data.giskard.allow``).
    key : str, optional
        JSONPath into the trace for the OPA input document
        (default: ``trace.last.outputs``).
    data : dict, optional
        Static data document merged via ``engine.add_data``
        (default: empty dict).

    Examples
    --------
    >>> from giskard.checks import RegoPolicy
    >>> check = RegoPolicy(
    ...     policy='''
    ... package giskard
    ...
    ... default allow = false
    ...
    ... allow if {
    ...     input.role == "admin"
    ... }
    ... ''',
    ...     rule="data.giskard.allow",
    ... )
    """

    policy: str = Field(..., description="Inline Rego policy source.")
    rule: str = Field(
        ...,
        description="Fully qualified boolean rule path (e.g. data.giskard.allow).",
    )
    key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the OPA input document.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Static data document merged into the policy engine.",
    )

    _engine: Any | None = PrivateAttr(default=None)

    def _compile_engine(self, regorus: Any) -> Any:
        """Load policy and data into a regorus engine and cache it."""
        engine = regorus.Engine()  # pyright: ignore[reportAttributeAccessIssue]
        engine.add_policy(_POLICY_FILENAME, self.policy)
        if self.data:
            engine.add_data(self.data)
        self._engine = engine
        return engine

    def _get_engine(self, regorus: Any) -> Any:
        if self._engine is None:
            return self._compile_engine(regorus)
        return self._engine

    @model_validator(mode="after")
    def _validate_rule_path(self) -> Self:
        """Validate the rule path and (optionally) the policy syntax.

        If `regorus` is installed, this validator parses the provided policy (and
        loads `data`) to surface syntax errors at instantiation time. If `regorus`
        is not installed, validation is skipped and a warning is emitted; runtime
        evaluation will still fail if `regorus` is missing.
        """
        if not self.rule.startswith("data."):
            raise ValueError("Rule path must start with 'data.'.")

        try:
            regorus = importlib.import_module("regorus")
        except ImportError:
            warnings.warn(
                _VALIDATION_SKIPPED_WARNING,
                category=RuntimeWarning,
                stacklevel=2,
            )
            return self

        try:
            self._compile_engine(regorus)
        except RuntimeError as err:
            raise ValueError(f"Invalid Rego policy or data: {err}") from err

        return self

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Execute the Rego policy check.

        Parameters
        ----------
        trace : Trace
            The trace containing interaction history. Access the current
            interaction via ``trace.last`` (preferred) or
            ``trace.interactions[-1]`` if available.

        Returns
        -------
        CheckResult
            Pass if the rule evaluates to ``True``, fail if ``False`` or
            undefined, error if regorus is missing, evaluation fails, or the
            rule value is not boolean.
        """
        raw_value = resolve(trace, self.key)
        details: dict[str, Any] = {
            "key": self.key,
            "data": self.data,
            "rule": self.rule,
        }

        if isinstance(raw_value, NoMatch):
            details["input"] = raw_value
            return CheckResult.failure(
                message=f"No value found for key '{self.key}'.",
                details=details,
            )

        details["input"] = raw_value

        try:
            input_json = json.dumps(raw_value)
        except (TypeError, ValueError) as err:
            details["error"] = str(err)
            return CheckResult.failure(
                message=f"Value at key '{self.key}' is not JSON serializable: {err}",
                details=details,
            )

        try:
            regorus = importlib.import_module("regorus")
        except ImportError:
            return CheckResult.error(
                message=_REGORUS_INSTALL_HINT,
                details=details,
            )

        try:
            engine = self._get_engine(regorus)
            engine.set_input_json(input_json)
            rule_value = engine.eval_rule(self.rule)
        except RuntimeError as err:
            details["error"] = str(err)
            return CheckResult.error(
                message=f"Rego policy evaluation failed: {err}",
                details=details,
            )

        return _result_from_boolean_rule(
            rule_value,
            rule=self.rule,
            details=details,
        )
