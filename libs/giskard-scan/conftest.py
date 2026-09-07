"""Keep optional-extra integration ``src`` trees out of collection when the extra is absent.

Why this file exists
--------------------
``addopts`` used to carry ``--ignore=src/giskard/scan/integrations/{garak,lidar,deepteam}``,
but pytest resolves a *relative* ``--ignore`` against the invocation CWD rather than the
rootdir. Those flags therefore only bit when pytest was started from inside
``libs/giskard-scan``. Run from the repo root (``pytest libs/giskard-scan``, which is what
``make test-unit-minimal`` and ``make test-garak`` both do) they resolved to nothing, the
ignores silently missed, and ``--doctest-modules`` imported every ``src`` module --
``ModuleNotFoundError: No module named 'garak'``.

``collect_ignore_glob`` is resolved relative to *this conftest's own directory*, so the
rule now holds for every invocation shape: repo root, lib chdir, a direct ``src`` path, or
an IDE runner. The three ``--ignore`` flags have been dropped from ``addopts`` because this
fully subsumes them -- keeping both would leave two mechanisms for one policy, one of them
CWD-dependent and silently broken.

The gate is conditional, which the unconditional ``--ignore`` flags could never express:
skip a tree only when its extra is *missing*. That is what lets one rule serve both
``make test-unit`` (extra absent -> skip) and ``make test-garak`` / ``test-lidar`` /
``test-deepteam`` (extra installed -> collect and run).

Only ``src`` trees are listed. The matching ``tests/integrations/<extra>`` trees must keep
collecting in every environment: they assert lazy-import safety and fail-fast behaviour and
are written to pass with the extra absent.

Notes for future maintainers
----------------------------
- ``find_spec`` is an *availability* oracle, not an importability check: a half-installed
  extra makes it return non-None, this file declines to ignore, and collection fails on the
  real import error. That is deliberate -- it is the same oracle used by ``_adapter.py``'s
  ``garak_available()``, by giskard-checks' ``require_optional_dependency``, and by the
  ``block_garak_import`` fixture, which monkeypatches ``find_spec`` specifically.
- This assumes each extra's *import name* equals its *directory name* under
  ``integrations/``. True for garak, lidar and deepteam today; an integration whose import
  name diverges from its directory would be silently mis-gated in either direction.
- Root cause behind the blast radius: ``--doctest-modules`` imports every ``src`` module,
  bypassing the package's lazy boundary (``__init__`` -> ``_adapter.py``, which imports
  ``TargetGenerator`` inside a function). The garak modules are exceptional because a
  *base class* cannot be deferred into a function body the way ``_adapter.py`` defers its
  imports. Narrowing ``--doctest-modules`` to modules that actually have doctests (the
  integration trees currently have none) is the real long-term fix and is tracked
  separately -- it is a larger change than this one.
"""

import importlib.util

# Extras whose src tree needs its optional dependency importable at module scope.
_OPTIONAL_EXTRAS = ("garak", "lidar", "deepteam")

collect_ignore_glob: list[str] = [
    f"src/giskard/scan/integrations/{_extra}/*"
    for _extra in _OPTIONAL_EXTRAS
    if importlib.util.find_spec(_extra) is None
]
