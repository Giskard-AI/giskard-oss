# Auto-register available adapters (silently skip if optional dep is missing)
try:
    from .garak import adapter as _garak_adapter  # noqa: F401
except ImportError:
    pass
