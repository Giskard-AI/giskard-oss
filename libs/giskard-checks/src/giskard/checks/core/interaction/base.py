from collections.abc import AsyncGenerator
from typing import ClassVar

from giskard.core import Discriminated, discriminated_base
from pydantic import ConfigDict
from pydantic.experimental.missing_sentinel import MISSING

from ..types import Target
from .interaction import Interaction
from .trace import Trace


@discriminated_base
class InteractionSpec[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Discriminated
):
    """Base class for interaction specifications that generate interactions.

    An interaction spec produces one or more `Interaction` objects by yielding
    them through an async generator. Each yielded interaction receives the updated
    trace (including the newly yielded interaction) via `generator.asend()`.

    This allows for multi-turn interactions where subsequent inputs can depend
    on the accumulated trace history.

    Subclasses must implement `generate()` to produce interactions. They should
    be registered using `@InteractionSpec.register("kind")` for polymorphic
    serialization.

    Attributes
    ----------
    InputType : TypeVar
        Type of the input values for interactions
    OutputType : TypeVar
        Type of the output values for interactions

    Notes
    -----
    Unknown fields are rejected (``extra="forbid"``), matching ``Check``.
    Interaction specs sit directly in persisted scenario JSON, so without this
    pydantic silently drops unrecognized keys: a scenario writing ``output``
    instead of ``outputs`` would leave ``outputs`` at its ``MISSING`` default
    and bind the scenario-level target instead of the value that was written.
    """

    # Rationale and the subclass rule: see ``Discriminated`` in giskard-core.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        """Generate interactions from the current trace state.

        This method is called by the scenario runner to produce interactions.
        It yields `Interaction` objects and receives updated traces (including
        the newly yielded interaction) via the async generator protocol.

        Parameters
        ----------
        trace : TraceType
            The current trace state before generating the interaction.

        Yields
        ------
        Interaction[InputType, OutputType]
            An interaction record to add to the trace.

        Receives
        --------
        TraceType
            The updated trace after the yielded interaction was added.
            Use `generator.asend(updated_trace)` to receive this value.

        Examples
        --------
        ```python
        async def generate(self, trace: TraceType) -> AsyncGenerator[Interaction, TraceType]:
            record = Interaction(inputs="hello", outputs="hi")
            updated_trace = yield record

            next_input = f"Previous had {len(updated_trace.interactions)} interactions"
            record = Interaction(inputs=next_input, outputs="response")
            yield record
        ```
        """
        raise NotImplementedError

    def bind_target(
        self,
        target: Target[InputType, OutputType, TraceType] | MISSING,
    ) -> "InteractionSpec[InputType, OutputType, TraceType]":
        """Bind a run/scenario target to this spec, returning a bound copy.

        Called by the scenario runner when a target is available (set at the
        scenario or ``run()`` level). Specs whose interactions need the target
        to fill in ``MISSING`` outputs override this to return a copy with the
        target applied; specs that resolve their own outputs override this as
        a no-op (the default) and return ``self`` unchanged.

        The default implementation is a no-op, so custom ``InteractionSpec``
        subclasses that do not need target binding are unaffected. Subclasses
        that *do* need it (notably ``Interact``) override this method to get
        the same hook the runner previously reserved for ``Interact`` via an
        ``isinstance`` special case.

        Parameters
        ----------
        target : Target | MISSING
            The SUT to bind. The runner only calls this with a concrete
            target (never ``MISSING``); the ``MISSING`` arm of the type keeps
            the signature consistent with the ``Target`` alias.

        Returns
        -------
        InteractionSpec[InputType, OutputType, TraceType]
            A spec with the target applied. The default returns ``self``.
        """
        return self
