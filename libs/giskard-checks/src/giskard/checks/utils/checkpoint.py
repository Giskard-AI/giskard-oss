"""Append-only JSONL checkpoint store for suite generate/run resume.

By default, checkpoints land under ``.giskard/checkpoints/<fingerprint>/`` and
matching runs resume automatically. Pass ``checkpoint_dir=False`` (or set
``GISKARD_CHECKPOINT=0``) to disable. Checkpoints may contain prompts, traces,
and model outputs — keep them local and out of version control.
"""

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.jsonl"
CHECKPOINT_ID_KEY = "checkpoint_id"

DEFAULT_CHECKPOINT_ROOT = Path(".giskard/checkpoints")
CHECKPOINT_DIR_ENV = "GISKARD_CHECKPOINT_DIR"
CHECKPOINT_RESUME_ENV = "GISKARD_CHECKPOINT_RESUME"
CHECKPOINT_ENABLED_ENV = "GISKARD_CHECKPOINT"

type ResumeMode = bool | Literal["force"]
type CheckpointDirArg = Path | str | bool | None


def ensure_checkpoint_id(
    annotations: dict[str, Any],
    *,
    name: str,
    index: int,
    suite_name: str,
) -> str:
    """Return a stable checkpoint id, assigning one into ``annotations`` if needed.

    Parameters
    ----------
    annotations : dict[str, Any]
        Scenario annotations dict (mutated when an id is assigned).
    name : str
        Scenario name.
    index : int
        Zero-based position in the suite.
    suite_name : str
        Suite name used to namespace the id.

    Returns
    -------
    str
        Stable id for resume skip lists.
    """
    existing = annotations.get(CHECKPOINT_ID_KEY)
    if existing is not None:
        return str(existing)
    digest = hashlib.sha256(f"{suite_name}\0{index}\0{name}".encode()).hexdigest()[:16]
    annotations[CHECKPOINT_ID_KEY] = digest
    return digest


def run_fingerprint(suite_name: str, scenario_ids: list[str]) -> dict[str, Any]:
    """Build the manifest fingerprint for a suite run phase."""
    return {
        "phase": "run",
        "suite": suite_name,
        "scenario_ids": list(scenario_ids),
    }


def generate_fingerprint(
    *,
    description: str,
    languages: list[str],
    generator_keys: list[str],
    seed: int,
    target_mode: str,
    max_scenarios: int | None,
) -> dict[str, Any]:
    """Build the manifest fingerprint for a suite generation phase."""
    return {
        "phase": "generate",
        "description": description,
        "languages": list(languages),
        "generators": list(generator_keys),
        "seed": seed,
        "target_mode": target_mode,
        "max_scenarios": max_scenarios,
    }


def generator_checkpoint_key(generator: Any, index: int) -> str:
    """Stable key for a generator instance within a generate_suite call."""
    cls = type(generator)
    base = f"{cls.__module__}.{cls.__name__}"
    name = getattr(generator, "name", None)
    if name is not None:
        return f"{base}:{name}:{index}"
    return f"{base}:{index}"


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    """Short stable directory name for a fingerprint dict."""
    raw = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def store_path_for(root: Path, fingerprint: dict[str, Any]) -> Path:
    """Return ``root / <fingerprint_hash>`` for an isolated run store."""
    return root / fingerprint_hash(fingerprint)


class CheckpointFingerprintError(ValueError):
    """Raised when an existing checkpoint fingerprint does not match."""


def _env_checkpoint_disabled() -> bool:
    value = os.environ.get(CHECKPOINT_ENABLED_ENV, "").strip().lower()
    return value in {"0", "false", "no", "off"}


def resolve_checkpoint_options(
    checkpoint_dir: CheckpointDirArg = None,
    resume: ResumeMode | None = None,
    *,
    fingerprint: dict[str, Any],
) -> tuple[Path | None, ResumeMode]:
    """Resolve store path and resume mode from API args, fingerprint, and env.

    Default behavior (no args, checkpointing enabled): write under
    ``.giskard/checkpoints/<fingerprint_hash>/`` (or ``GISKARD_CHECKPOINT_DIR``
    as the root) and resume when that store already exists
    (``resume`` defaults to ``True``).

    Parameters
    ----------
    checkpoint_dir : Path | str | bool | None, optional
        ``None`` — auto root + fingerprint subdir.
        ``False`` — disable checkpointing.
        ``Path`` / ``str`` — use as root; store is ``root / <fingerprint_hash>``.
    resume : bool | Literal[\"force\"] | None, optional
        ``None`` defaults to ``True`` (or env). ``False`` starts a fresh store.
    fingerprint : dict[str, Any]
        Run identity used for the subdir name and manifest.

    Returns
    -------
    tuple[Path | None, bool | Literal[\"force\"]]
        Concrete store directory (or ``None`` if disabled) and resume mode.
    """
    if checkpoint_dir is False or _env_checkpoint_disabled():
        return None, False

    if isinstance(checkpoint_dir, bool):
        raise TypeError("checkpoint_dir must be a path, None, or False")

    if checkpoint_dir is None:
        env_dir = os.environ.get(CHECKPOINT_DIR_ENV)
        root = Path(env_dir) if env_dir else DEFAULT_CHECKPOINT_ROOT
    else:
        root = Path(checkpoint_dir)

    path = store_path_for(root, fingerprint)

    if resume is None:
        env_resume = os.environ.get(CHECKPOINT_RESUME_ENV, "").strip().lower()
        if env_resume == "force":
            resolved_resume: ResumeMode = "force"
        elif env_resume in {"0", "false", "no", "off"}:
            resolved_resume = False
        elif env_resume in {"1", "true", "yes"}:
            resolved_resume = True
        else:
            resolved_resume = True
    else:
        resolved_resume = resume

    return path, resolved_resume


class RunStore:
    """Durable append-only event store for eval checkpoints.

    Parameters
    ----------
    path : Path
        Directory containing ``manifest.json`` and ``events.jsonl``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._events_path = path / EVENTS_NAME
        self._manifest_path = path / MANIFEST_NAME
        self._lock = asyncio.Lock()
        self._events: list[dict[str, Any]] = []

    @classmethod
    async def open(
        cls,
        path: Path | str,
        *,
        fingerprint: dict[str, Any],
        resume: ResumeMode = False,
    ) -> "RunStore":
        """Create or open a checkpoint directory.

        Parameters
        ----------
        path : Path | str
            Checkpoint directory.
        fingerprint : dict[str, Any]
            Run identity written to ``manifest.json``. Must match on resume
            unless ``resume=\"force\"``.
        resume : bool | Literal[\"force\"]
            If ``False`` and the directory already has a manifest, overwrite
            with a fresh store. If ``True``, require fingerprint match. If
            ``\"force\"``, load existing events even when fingerprints differ.

        Returns
        -------
        RunStore
            Open store ready for append / load.

        Raises
        ------
        CheckpointFingerprintError
            When ``resume=True`` and the stored fingerprint differs.
        """
        directory = Path(path)
        store = cls(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if store._manifest_path.exists() and resume:
            stored = json.loads(store._manifest_path.read_text())
            stored_fp = stored.get("fingerprint", {})
            if resume != "force" and stored_fp != fingerprint:
                raise CheckpointFingerprintError(
                    f"Checkpoint fingerprint mismatch at {directory}: "
                    f"stored={stored_fp!r} requested={fingerprint!r}. "
                    "Pass resume='force' to override."
                )
            store._reload_events()
        else:
            store._write_manifest(fingerprint)
            store._events_path.write_text("")
            store._events = []

        return store

    def _write_manifest(self, fingerprint: dict[str, Any]) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "fingerprint": fingerprint,
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def _reload_events(self) -> None:
        self._events = []
        if not self._events_path.exists():
            return
        for line in self._events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            self._events.append(json.loads(line))

    async def append(
        self,
        event_type: str,
        *,
        id: str,
        payload: dict[str, Any],
    ) -> None:
        """Append one event record and flush to disk.

        Parameters
        ----------
        event_type : str
            Event kind (e.g. ``scenario_finished``, ``scenario_generated``).
        id : str
            Stable unit id used for resume skip lists.
        payload : dict[str, Any]
            JSON-serializable event body.
        """
        record = {"type": event_type, "id": id, "payload": payload}
        async with self._lock:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._events.append(record)

    def completed_ids(self, event_type: str) -> set[str]:
        """Return ids that have at least one event of ``event_type``."""
        return {
            event["id"] for event in self._events if event.get("type") == event_type
        }

    def load_payloads(self, event_type: str) -> dict[str, dict[str, Any]]:
        """Return the latest payload per id for ``event_type``."""
        payloads: dict[str, dict[str, Any]] = {}
        for event in self._events:
            if event.get("type") == event_type:
                payloads[event["id"]] = event["payload"]
        return payloads
