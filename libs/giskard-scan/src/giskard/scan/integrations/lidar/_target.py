"""Bridge a giskard.scan Target into a lidar Target (a giskard.agents BaseGenerator).

``BaseGenerator`` is the workspace ``giskard.agents`` type and is imported eagerly
(no lidar needed). lidar's compat types (``make_response``) are imported LAZILY
inside ``_call_model`` so ``import giskard.scan.integrations.lidar`` works without
lidar installed. lidar adapts the workspace ``giskard.agents`` via its own
``giskard_compat`` layer — there is no second copy of ``giskard.agents``.
"""

import uuid
from typing import Any

# BaseGenerator is the workspace giskard.agents type — always importable, no lidar needed.
from giskard.agents import BaseGenerator

# Message/Response/make_response come from lidar's compat layer — import LAZILY inside
# _call_model (below), NOT here, so this module imports without lidar installed.
# These ARE the workspace giskard.checks (the scan side of the bridge).
from giskard.checks import DatasetInputGenerator, Interact, Interaction, Target, Trace
from pydantic import PrivateAttr


class ScanTargetGenerator(BaseGenerator):
    """Presents a scan Target to lidar as a lidar Target.

    Subclasses lidar's BaseGenerator (Pydantic) so it satisfies lidar's
    runtime_checkable Target protocol, which the scanner probes with isinstance
    and calls model_copy(update={"middlewares": ...}) / model_dump on. A plain
    wrapper cannot satisfy that; a BaseGenerator subclass gets those for free.

    Statefulness via thread_id round-trip (mirrors the app's HubTarget):
    multiturn probes negotiate session state through Response.metadata["thread_id"].
    We keep one accumulating scan Trace per thread_id in ``_threads`` and always
    return the thread_id, so the probe can send only the latest turn next time and
    still have the Target driven with full accumulated context. This mirrors garak's
    TargetGenerator.internal_cache, keyed on metadata["thread_id"] instead of
    Message.notes["uuid"].
    """

    model_config = {"arbitrary_types_allowed": True}

    # The scan Target is a PRIVATE attr, not a pydantic field: lidar's scanner
    # serializes the generator via model_dump(mode="json") for run metadata, and a
    # scan Target (often a plain function) is not JSON-serializable. A PrivateAttr is
    # excluded from model_dump, so the scan runs; model_copy(update={"middlewares":
    # ...}) still preserves it because model_copy carries private attrs over.
    _scan_target: Target = PrivateAttr()  # pyright: ignore[reportMissingTypeArgument]

    # Per-thread trace cache. default_factory gives each instance its own dict
    # (not a shared mutable class-level default).
    _threads: dict[str, Trace] = PrivateAttr(default_factory=dict)  # pyright: ignore[reportMissingTypeArgument]

    # Structured interactions this bridge built, keyed by lidar's per-call
    # call_id (from get_current_target_call_id()). The adapter joins
    # attempt.target_calls back to these to rebuild a lossless display Trace,
    # instead of reconstructing it from flattened attempt.messages.
    _by_call_id: dict[str, Interaction] = PrivateAttr(default_factory=dict)  # pyright: ignore[reportMissingTypeArgument]

    def __init__(self, target: Target, **data: Any):  # pyright: ignore[reportMissingTypeArgument]
        super().__init__(**data)
        self._scan_target = target

    async def _call_model(
        self,
        messages: Any,
        params: Any = None,
        metadata: "dict[str, Any] | None" = None,
    ) -> Any:  # -> lidar.giskard_compat.Response
        # Lazy: lidar's compat layer + call-id side channel, only importable when
        # lidar is installed. get_current_target_call_id() returns the in-flight
        # call id set by lidar's TargetCallTrackingMiddleware for THIS complete().
        from lidar.core.models.target import get_current_target_call_id
        from lidar.giskard_compat import make_response

        thread_id = (metadata or {}).get("thread_id") or str(uuid.uuid4())
        # Accumulated trace for this thread (empty on first turn). Trace is frozen,
        # so with_interaction returns a NEW trace we write back under the same key.
        trace = self._threads.get(thread_id) or Trace.for_target(self._scan_target)

        text = messages[-1].content if messages else ""
        interaction = Interact(
            inputs=DatasetInputGenerator(prompt=text),
            outputs=self._scan_target,
        )
        trace = await trace.with_interaction(interaction)
        self._threads[thread_id] = trace

        # Keep the typed Interaction we just built, keyed by lidar's call_id, so
        # the adapter can join attempt.target_calls back to it at result time and
        # avoid reconstructing a lossy trace from flattened messages. call_id is
        # None only outside a tracked complete() (e.g. discovery) — skip then.
        call_id = get_current_target_call_id()
        if call_id is not None and trace.last is not None:
            self._by_call_id[call_id] = trace.last

        outputs = trace.last.outputs if trace.last is not None else None
        reply = str(outputs) if outputs is not None else ""
        # Round-trip the thread_id so the probe threads subsequent turns.
        # make_response builds a lidar.giskard_compat.Response (subclass of
        # giskard's CompletionResponse) that carries a settable .metadata dict.
        return make_response(
            role="assistant", content=reply, metadata={"thread_id": thread_id}
        )
