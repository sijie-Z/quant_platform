"""Utility decorators: timing, validation, caching."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from quant_platform.utils.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")


def timer(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log function execution time."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger()
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("%s took %.3fs", func.__name__, elapsed)

    return wrapper


def validate_output(validator: Callable[[Any], None]):
    """Decorator factory: validates function output with a validator function.

    Usage:
        @validate_output(lambda df: assert not df.isnull().any().any())
        def compute_something(...): ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            validator(result)
            return result

        return wrapper

    return decorator


def cached_property_with_ttl(ttl_seconds: float = 3600):
    """Decorator for computed properties with a simple TTL cache.

    For read-heavy, compute-light properties that don't change within a session.
    """

    def decorator(func: Callable[[Any], R]) -> property:
        cache_key = f"__cached_{func.__name__}"

        def wrapper(self: Any) -> R:
            cached = getattr(self, cache_key, None)
            if cached is not None:
                value, timestamp = cached
                if time.monotonic() - timestamp < ttl_seconds:
                    return value
            result = func(self)
            setattr(self, cache_key, (result, time.monotonic()))
            return result

        return property(functools.wraps(func)(wrapper))

    return decorator
