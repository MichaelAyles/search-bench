"""Retry helpers with exponential backoff for transient errors."""

import asyncio
from typing import Callable, Awaitable, TypeVar

from ..wrappers.base import Query, QueryResult, SearchMode

# ---------------------------------------------------------------------------
# Retry classification patterns
# ---------------------------------------------------------------------------

_RETRYABLE = ("timeout", "429", "rate limit", "rate_limit", "too many", "503", "overloaded", "connection")
_RATE_LIMIT = ("429", "rate limit", "rate_limit", "too many")


# ---------------------------------------------------------------------------
# Generic retry core
# ---------------------------------------------------------------------------

T = TypeVar("T")


async def _retry_loop(
    callable_fn: Callable[[], Awaitable[tuple[T, str | None]]],
    label: str,
    max_retries: int = 3,
) -> tuple[T, str | None, int, float]:
    """Generic retry loop with exponential backoff.

    Args:
        callable_fn: Async callable returning (result, error_or_None).
        label: Display name for retry log messages.
        max_retries: Maximum number of retries.

    Returns:
        (result, error, retry_count, rate_limit_wait_seconds).
    """
    delay = 5.0
    rate_limit_wait = 0.0

    for attempt in range(max_retries + 1):
        result, error = await callable_fn()

        if error is None:
            return result, None, attempt, rate_limit_wait

        if attempt >= max_retries:
            break

        err_lower = error.lower()
        is_rate_limit = any(x in err_lower for x in _RATE_LIMIT)
        is_retryable = any(x in err_lower for x in _RETRYABLE)

        if not is_retryable:
            break

        wait = delay * 2 if is_rate_limit else delay
        if is_rate_limit:
            rate_limit_wait += wait

        print(
            f"\n  [retry {attempt + 1}/{max_retries}] {label} "
            f"error: {error[:60]} — retrying in {wait:.0f}s"
        )
        await asyncio.sleep(wait)
        delay = min(delay * 2, 60.0)

    return result, error, attempt, rate_limit_wait


# ---------------------------------------------------------------------------
# Read-only query retry wrapper
# ---------------------------------------------------------------------------

async def _run_with_retry(
    wrapper,
    query: Query,
    mode: SearchMode,
    run_number: int,
    max_retries: int = 3,
) -> tuple[QueryResult, int, float]:
    """Run a read-only query with exponential backoff retry.

    Returns (result, retry_count, total_rate_limit_wait_seconds).
    Only retries on timeout / rate-limit / transient server errors.
    """

    async def _call() -> tuple[QueryResult, str | None]:
        r = await wrapper.run_query(query, mode, run_number)
        return r, r.error

    result, _error, retry_count, rl_wait = await _retry_loop(
        _call, wrapper.name(), max_retries=max_retries
    )
    return result, retry_count, rl_wait


# ---------------------------------------------------------------------------
# Tool task retry wrapper (author / review phases)
# ---------------------------------------------------------------------------

async def _tool_with_retry(
    tool_name: str,
    prompt: str,
    codebase_dir: "Path",
    max_retries: int = 3,
    timeout: int = 180,
    *,
    _run_tool_for_task: Callable | None = None,
) -> tuple[str, str | None, int, float]:
    """Run _run_tool_for_task with retries.

    Returns (output, error, retry_count, rate_limit_wait_seconds).

    The _run_tool_for_task callable is injected to avoid circular imports.
    """
    # Import lazily to avoid circular dependency
    if _run_tool_for_task is None:
        from .runner import _run_tool_for_task as _fn
        _run_tool_for_task = _fn

    last_output = ""

    async def _call() -> tuple[str, str | None]:
        nonlocal last_output
        output, error = await _run_tool_for_task(tool_name, prompt, codebase_dir, timeout)
        last_output = output
        return output, error

    result, error, retry_count, rl_wait = await _retry_loop(
        _call, tool_name, max_retries=max_retries
    )
    return result, error, retry_count, rl_wait
