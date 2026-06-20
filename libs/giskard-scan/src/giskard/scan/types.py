from typing import TypedDict


class ScanOptions(TypedDict, total=False):
    """Optional keyword settings for a vulnerability scan.

    Passed to :func:`~giskard.scan.vulnerability.vulnerability_scan` as unpacked
    keyword arguments. All keys are optional; omitted keys fall back to the
    defaults documented on each field.

    Attributes:
        max_scenarios: Total upper bound on scenarios across all vulnerability
            generators. ``None`` lets each generator apply its own default.
            Defaults to ``None``.
        seed: Integer seed used for reproducible scenario generation.
            Defaults to ``42``.
        group_by: Result annotation key used to group the printed report.
            ``None`` prints the ungrouped report. Defaults to ``"threat-type"``.
        parallel: When ``True``, run scenarios concurrently. Defaults to ``True``.
        max_concurrency: Cap on concurrent scenarios when ``parallel=True``.
            ``None`` runs all scenarios at once. Defaults to ``None``.
        commercial_use: When ``True``, exclude generators whose datasets do not
            permit commercial use. Defaults to ``False``.
    """

    max_scenarios: int | None
    seed: int
    group_by: str | None
    parallel: bool
    max_concurrency: int | None
    commercial_use: bool


class ResolvedScanOptions(TypedDict):
    """``ScanOptions`` with every key resolved (defaults merged in).

    Produced internally by merging caller overrides over the defaults, so all
    keys are present and can be accessed without ``reportTypedDictNotRequiredAccess``.

    Note: this mirrors :class:`ScanOptions`'s fields rather than subclassing it,
    because ``total=True`` on a subclass does not re-mark keys inherited from a
    ``total=False`` parent as required (PEP 655 — ``total`` only sets the default
    for keys declared in the class body).
    """

    max_scenarios: int | None
    seed: int
    group_by: str | None
    parallel: bool
    max_concurrency: int | None
    commercial_use: bool
