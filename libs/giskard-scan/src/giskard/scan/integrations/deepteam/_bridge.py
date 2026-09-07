"""Bridge a giskard.scan Target into DeepTeam's ``model_callback``.

DeepTeam calls the callback ``(input, turns)`` live, once per turn, with an
accumulating ``turns`` list, and appends the exact ``RTTurn`` we return
unmodified. We stamp a uuid into that ``RTTurn.metadata`` and key a per-
conversation typed ``Trace`` cache on it (garak's ``internal_cache`` pattern),
so multi-turn conversations accumulate a lossless typed Trace instead of
degrading to strings. ``giskard.checks`` is always importable; ``RTTurn`` is
imported lazily so this module imports without deepteam installed.
"""

import uuid as uuid_module
from typing import Any

from giskard.checks import (
    DatasetInputGenerator,
    Interact,
    Target,
    Trace,
)


class ScanTargetCallback:
    """Present a scan ``Target`` to DeepTeam as an async ``model_callback``.

    ``traces`` maps our per-conversation uuid to the accumulating frozen
    ``Trace`` the adapter reads at result time.
    """

    METADATA_UUID_KEY = "giskard_uuid"

    def __init__(self, target: Target) -> None:  # pyright: ignore[reportMissingTypeArgument]
        self._target = target
        self.traces: dict[str, Trace] = {}  # pyright: ignore[reportMissingTypeArgument]

    def _recover_uuid(self, turns: "list[Any] | None") -> "str | None":
        """Return the uuid stamped on the newest prior assistant turn, if any.

        Linear multi-turn attacks (linear/crescendo/bad-likert/sequential) thread
        the same ``turns`` list forward, so our stamped assistant turn is always
        present and the whole conversation accumulates under one uuid.

        Tree-search attacks (``TreeJailbreaking``) are the exception: they explore
        branches purely in the simulator's own JSON space and never call this
        callback while branching, then *replay* the winning path against the
        callback starting from a fresh user-only ``turns`` list. That replay mints
        a new uuid, so a tree attack can leave a couple of short-lived partial
        traces in ``traces`` for one conversation. This causes mild, transient
        cache growth (freed when the scan ends) and never a uuid collision or a
        corrupted trace — divergent branches never share a uuid because branching
        never reaches us. The final ``RTTestCase.turns`` the adapter reads stays
        coherent. See spec POC notes / Task 4 review.
        """
        for turn in reversed(turns or []):
            metadata = getattr(turn, "metadata", None)
            if metadata and metadata.get(self.METADATA_UUID_KEY):
                return metadata[self.METADATA_UUID_KEY]
        return None

    def trace_for(self, turns: "list[Any] | None") -> "Trace | None":  # pyright: ignore[reportMissingTypeArgument]
        """Return the lossless Trace accumulated for *turns*, or None if we never
        produced a turn in this conversation (e.g. a seeded opening turn).

        The uuid cache is ours: callers hand us the test case's turns and get a
        Trace back rather than reaching into ``traces``/``METADATA_UUID_KEY``.
        """
        conversation_uuid = self._recover_uuid(turns)
        if conversation_uuid is None:
            return None
        return self.traces.get(conversation_uuid)

    async def __call__(self, input: str, turns: "list[Any] | None" = None) -> Any:
        from deepteam.test_case import RTTurn

        conversation_uuid = self._recover_uuid(turns) or str(uuid_module.uuid4())
        trace = self.traces.get(conversation_uuid) or Trace.for_target(self._target)

        interaction = Interact(
            inputs=DatasetInputGenerator(prompt=input),
            outputs=self._target,
        )
        # Trace is frozen: with_interaction drives the target and returns a NEW
        # trace, which we store back under the same uuid for the next turn.
        trace = await trace.with_interaction(interaction)
        self.traces[conversation_uuid] = trace

        outputs = trace.last.outputs if trace.last is not None else None
        reply = str(outputs) if outputs is not None else ""
        return RTTurn(
            role="assistant",
            content=reply,
            metadata={self.METADATA_UUID_KEY: conversation_uuid},
        )
