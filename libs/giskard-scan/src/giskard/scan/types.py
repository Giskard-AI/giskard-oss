from typing import TypedDict


class SharedScanOptions(TypedDict, total=False):
    """Optional execution settings shared by quality and vulnerability scans.

    Attributes:
        max_scenarios: Total upper bound on scenarios across all generators.
            ``None`` lets each generator apply its own default.
        seed: Integer seed used for reproducible scenario generation.
        group_by: Result tag key used to group the printed report.
            ``None`` prints the ungrouped report.
        parallel: When ``True``, run generated scenarios concurrently against
            the target (suite *execution*). Defaults to ``True`` for scans.
            Distinct from scenario *generation*, which always runs generators
            concurrently in :func:`~giskard.scan.catalog.generate_suite`.
            Pass ``False`` to force serial execution. Note that
            :meth:`~giskard.checks.scenarios.suite.Suite.run` itself defaults
            to ``parallel=False``.
        max_concurrency: Cap on concurrent scenarios when ``parallel=True``.
            ``None`` runs all scenarios at once (provider rate limits become
            the effective cap). When ``parallel=False``, a valid value has no
            effect on scheduling, but invalid values are still rejected.
            Defaults to ``None``.
        return_exception: When ``True``, a scenario whose input generation fails
            is recorded as an errored result and the scan continues. When
            ``False``, the failure propagates and aborts the scan.
    """

    max_scenarios: int | None
    seed: int
    group_by: str | None
    parallel: bool
    max_concurrency: int | None
    return_exception: bool


class ScanOptions(SharedScanOptions, total=False):
    """Optional keyword settings for a vulnerability scan.

    Extends :class:`SharedScanOptions` with vulnerability-only keys. Prefer
    passing these keys as explicit keyword arguments to
    :func:`~giskard.scan.vulnerability.vulnerability_scan`.

    Attributes:
        commercial_use: When ``True``, exclude generators whose datasets do not
            permit commercial use. Defaults to ``False``.
    """

    commercial_use: bool


class ResolvedSharedScanOptions(TypedDict):
    """Shared scan options with every key resolved.

    Produced by :func:`resolve_scan_options` so all keys are present and can
    be accessed without ``reportTypedDictNotRequiredAccess``.
    """

    max_scenarios: int | None
    seed: int
    group_by: str | None
    parallel: bool
    max_concurrency: int | None
    return_exception: bool


class ResolvedScanOptions(ResolvedSharedScanOptions):
    """Vulnerability scan options with every key resolved.

    Note: this mirrors :class:`ScanOptions`'s fields rather than subclassing
    it, because ``total=True`` on a subclass does not re-mark keys inherited
    from a ``total=False`` parent as required (PEP 655 — ``total`` only sets
    the default for keys declared in the class body).
    """

    commercial_use: bool


def resolve_scan_options(
    *,
    max_scenarios: int | None,
    seed: int,
    group_by: str | None,
    parallel: bool,
    max_concurrency: int | None,
    return_exception: bool,
    commercial_use: bool = False,
) -> ResolvedScanOptions:
    """Resolve explicit scan keyword arguments into a complete options dict.

    Parameters
    ----------
    max_scenarios : int or None
        Total upper bound on scenarios across all generators.
    seed : int
        Integer seed used for reproducible scenario generation.
    group_by : str or None
        Result tag key used to group the printed report.
    parallel : bool
        When ``True``, run generated scenarios concurrently against the target.
        This is suite *execution*; scenario *generation* always runs generators
        concurrently in :func:`~giskard.scan.catalog.generate_suite`.
    max_concurrency : int or None
        Cap on concurrent scenarios when ``parallel=True``. ``None`` runs all
        scenarios at once (provider rate limits become the effective cap).
        When ``parallel=False``, a valid value has no effect on scheduling,
        but invalid values are still rejected.
    return_exception : bool
        When ``True``, record input-generation failures as errored results.
    commercial_use : bool, optional
        When ``True``, exclude generators whose datasets do not permit
        commercial use. Defaults to ``False``.

    Returns
    -------
    ResolvedScanOptions
        Scan options with every key present.
    """
    return {
        "max_scenarios": max_scenarios,
        "seed": seed,
        "group_by": group_by,
        "parallel": parallel,
        "max_concurrency": max_concurrency,
        "return_exception": return_exception,
        "commercial_use": commercial_use,
    }
