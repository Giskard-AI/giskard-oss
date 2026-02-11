from .base import BaseRateLimiter
from .basic import BasicRateLimiter

RateLimiter = BasicRateLimiter

__all__ = [
    "RateLimiter",
    "BaseRateLimiter",
]
