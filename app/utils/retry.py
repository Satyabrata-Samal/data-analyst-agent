"""Retry and timeout utilities for LLM calls.

Provides ``with_retry``, a decorator with exponential backoff for transient
Anthropic API failures, and ``with_timeout``, an async context manager that
bounds execution time using ``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, ParamSpec, TypeVar, cast

import anthropic

from app.config.settings import settings
from app.utils.logger import get_logger, log_error

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_RETRYABLE_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.APIStatusError,
    asyncio.TimeoutError,
)


def _backoff_seconds(attempt: int) -> float:
    return settings.retry_backoff_seconds * (2**attempt)


def _raise_exhausted(last_exception: Exception) -> None:
    raise RuntimeError(
        f"LLM call failed after {settings.max_retries} attempts: {last_exception}"
    ) from last_exception


def _handle_failure(func_name: str, attempt: int, exc: Exception) -> None:
    log_error(logger, func_name, exc)
    if attempt < settings.max_retries - 1:
        time.sleep(_backoff_seconds(attempt))


async def _handle_failure_async(func_name: str, attempt: int, exc: Exception) -> None:
    log_error(logger, func_name, exc)
    if attempt < settings.max_retries - 1:
        await asyncio.sleep(_backoff_seconds(attempt))


def with_retry(func: Callable[P, R]) -> Callable[P, R]:
    """Retry a sync or async function on transient LLM failures."""

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Exception | None = None

            for attempt in range(settings.max_retries):
                try:
                    result = await cast(Callable[P, Awaitable[R]], func)(*args, **kwargs)
                    return result
                except Exception as exc:
                    last_exception = exc
                    await _handle_failure_async(func.__name__, attempt, exc)

            assert last_exception is not None
            _raise_exhausted(last_exception)

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        last_exception: Exception | None = None

        for attempt in range(settings.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                _handle_failure(func.__name__, attempt, exc)

        assert last_exception is not None
        _raise_exhausted(last_exception)

    return sync_wrapper


@asynccontextmanager
async def with_timeout(seconds: int) -> AsyncGenerator[None, None]:
    """Bound the wrapped async block to ``seconds`` using ``asyncio.wait_for``."""
    async with asyncio.timeout(seconds):
        yield
