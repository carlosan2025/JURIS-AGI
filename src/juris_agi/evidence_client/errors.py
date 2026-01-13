"""
Typed exceptions and retry/backoff logic for Evidence API client.
"""

import functools
import logging
import random
import time
from typing import Callable, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EvidenceAPIError(Exception):
    """Base exception for Evidence API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class EvidenceConnectionError(EvidenceAPIError):
    """Failed to connect to Evidence API."""

    pass


class EvidenceTimeoutError(EvidenceAPIError):
    """Request to Evidence API timed out."""

    pass


class EvidenceAuthenticationError(EvidenceAPIError):
    """Authentication failed (401)."""

    pass


class EvidenceAuthorizationError(EvidenceAPIError):
    """Authorization failed (403)."""

    pass


class EvidenceNotFoundError(EvidenceAPIError):
    """Resource not found (404)."""

    pass


class EvidenceValidationError(EvidenceAPIError):
    """Request validation failed (422)."""

    pass


class EvidenceRateLimitError(EvidenceAPIError):
    """Rate limit exceeded (429)."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        retry_after: Optional[float] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, status_code, details)
        self.retry_after = retry_after


class EvidenceServerError(EvidenceAPIError):
    """Server error (5xx)."""

    pass


class EvidenceUnavailableError(EvidenceAPIError):
    """Service unavailable - Evidence API is not configured or reachable."""

    pass


def classify_http_error(status_code: int, message: str, details: Optional[dict] = None) -> EvidenceAPIError:
    """Convert HTTP status code to appropriate exception type."""
    error_map: dict[int, Type[EvidenceAPIError]] = {
        401: EvidenceAuthenticationError,
        403: EvidenceAuthorizationError,
        404: EvidenceNotFoundError,
        422: EvidenceValidationError,
        429: EvidenceRateLimitError,
    }

    if status_code in error_map:
        return error_map[status_code](message, status_code, details)
    elif 500 <= status_code < 600:
        return EvidenceServerError(message, status_code, details)
    else:
        return EvidenceAPIError(message, status_code, details)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
        retryable_status_codes: Optional[set[int]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            EvidenceConnectionError,
            EvidenceTimeoutError,
            EvidenceServerError,
            EvidenceRateLimitError,
        )
        self.retryable_status_codes = retryable_status_codes or {429, 500, 502, 503, 504}

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate delay before next retry with exponential backoff."""
        if retry_after is not None:
            return min(retry_after, self.max_delay)

        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception and attempt count."""
        if attempt >= self.max_retries:
            return False

        if isinstance(exception, self.retryable_exceptions):
            return True

        if isinstance(exception, EvidenceAPIError) and exception.status_code:
            return exception.status_code in self.retryable_status_codes

        return False


def with_retry(config: Optional[RetryConfig] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e, attempt):
                        raise

                    # Get retry_after if available
                    retry_after = None
                    if isinstance(e, EvidenceRateLimitError):
                        retry_after = e.retry_after

                    delay = config.calculate_delay(attempt, retry_after)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )
                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper

    return decorator


async def with_retry_async(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs,
) -> T:
    """Async version of retry logic."""
    import asyncio

    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not config.should_retry(e, attempt):
                raise

            retry_after = None
            if isinstance(e, EvidenceRateLimitError):
                retry_after = e.retry_after

            delay = config.calculate_delay(attempt, retry_after)
            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {delay:.2f}s due to: {e}"
            )
            await asyncio.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")
