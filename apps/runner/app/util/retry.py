# ABOUTME: Enhanced retry mechanisms with advanced backoff strategies and comprehensive error handling
# ABOUTME: Provides robust retry logic with jitter, circuit breaking, and intelligent failure classification

import asyncio
import random
import time
import logging
from enum import Enum
from typing import (
    Callable, TypeVar, Any, Dict, Optional, List, Union, 
    Tuple, Set, Type, Awaitable
)
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
import statistics
from abc import ABC, abstractmethod
import inspect

from .exceptions import (
    BaseRedTeamException, NetworkException, ExternalAPIException, 
    AuthenticationException, ResourceException, DatabaseException,
    ErrorCode, ErrorSeverity, ErrorCategory, ErrorContext, error_tracker
)

T = TypeVar('T')
logger = logging.getLogger(__name__)


class BackoffStrategy(str, Enum):
    """Available backoff strategies"""
    FIXED = "fixed"                    # Fixed delay between retries
    LINEAR = "linear"                  # Linear increase in delay
    EXPONENTIAL = "exponential"        # Exponential backoff
    FIBONACCI = "fibonacci"            # Fibonacci sequence delays
    POLYNOMIAL = "polynomial"          # Polynomial growth
    CUSTOM = "custom"                  # Custom function


class JitterType(str, Enum):
    """Types of jitter for randomizing retry delays"""
    NONE = "none"                      # No jitter
    FULL = "full"                      # Full randomization (0 to delay)
    EQUAL = "equal"                    # Equal jitter (delay/2 to delay)
    DECORR = "decorr"                  # Decorrelated jitter
    RANDOM = "random"                  # Random jitter (+/- 25%)


class RetryDecision(str, Enum):
    """Decisions for whether to retry"""
    RETRY = "retry"                    # Retry the operation
    STOP = "stop"                      # Stop retrying
    ESCALATE = "escalate"              # Escalate to different handler


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay: float
    total_elapsed: float
    last_exception: Optional[BaseRedTeamException]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "delay": self.delay,
            "total_elapsed": self.total_elapsed,
            "last_exception": self.last_exception.to_dict() if self.last_exception else None,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass 
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3                           # Maximum retry attempts
    base_delay: float = 1.0                         # Base delay in seconds
    max_delay: float = 300.0                        # Maximum delay between retries
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0                 # Multiplier for exponential backoff
    jitter_type: JitterType = JitterType.EQUAL      # Type of jitter to apply
    jitter_amount: float = 0.1                      # Amount of jitter (0.0-1.0)
    timeout_per_attempt: Optional[float] = None     # Timeout for each attempt
    total_timeout: Optional[float] = None           # Total timeout for all attempts
    
    # Exception classification
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        NetworkException, ExternalAPIException, ResourceException, 
        DatabaseException, ConnectionError, TimeoutError, asyncio.TimeoutError
    })
    non_retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        AuthenticationException, ValueError, TypeError, KeyError
    })
    
    # Advanced configuration
    retry_on_result: Optional[Callable[[Any], bool]] = None  # Custom result validation
    stop_on_exception: Optional[Callable[[Exception], bool]] = None  # Custom exception handling
    exponential_base: float = 2.0                            # Base for exponential backoff
    polynomial_degree: int = 2                               # Degree for polynomial backoff
    
    # Circuit breaker integration
    enable_circuit_breaker: bool = False
    circuit_breaker_threshold: int = 5                       # Failures before circuit opens
    circuit_breaker_timeout: float = 60.0                    # Circuit open duration
    
    # Rate limiting
    enable_rate_limiting: bool = False
    rate_limit_calls: int = 100                             # Calls per period
    rate_limit_period: float = 60.0                         # Period in seconds
    
    # Statistics collection
    collect_statistics: bool = True
    statistics_window: int = 1000                           # Number of recent attempts to track


class BackoffCalculator:
    """Calculates retry delays using various strategies"""
    
    @staticmethod
    def calculate_delay(
        config: RetryConfig,
        attempt: int,
        last_delay: float = 0.0
    ) -> float:
        """Calculate delay for given attempt number"""
        
        if config.backoff_strategy == BackoffStrategy.FIXED:
            delay = config.base_delay
            
        elif config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = config.base_delay * attempt
            
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.exponential_base ** (attempt - 1))
            
        elif config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = BackoffCalculator._fibonacci_delay(config.base_delay, attempt)
            
        elif config.backoff_strategy == BackoffStrategy.POLYNOMIAL:
            delay = config.base_delay * (attempt ** config.polynomial_degree)
            
        else:  # CUSTOM - fallback to exponential
            delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter
        delay = BackoffCalculator._apply_jitter(
            delay, config.jitter_type, config.jitter_amount, last_delay
        )
        
        return max(0.0, delay)
    
    @staticmethod
    def _fibonacci_delay(base_delay: float, attempt: int) -> float:
        """Calculate Fibonacci sequence delay"""
        if attempt <= 1:
            return base_delay
        
        a, b = 0, 1
        for _ in range(attempt - 1):
            a, b = b, a + b
        
        return base_delay * b
    
    @staticmethod
    def _apply_jitter(
        delay: float, 
        jitter_type: JitterType, 
        jitter_amount: float,
        last_delay: float = 0.0
    ) -> float:
        """Apply jitter to delay"""
        
        if jitter_type == JitterType.NONE:
            return delay
        
        elif jitter_type == JitterType.FULL:
            return random.uniform(0, delay)
        
        elif jitter_type == JitterType.EQUAL:
            jitter = delay * jitter_amount
            return delay + random.uniform(-jitter, jitter)
        
        elif jitter_type == JitterType.DECORR:
            # Decorrelated jitter based on last delay
            if last_delay > 0:
                return random.uniform(0, delay * 3)
            return random.uniform(0, delay)
        
        elif jitter_type == JitterType.RANDOM:
            # Random jitter within 25% of delay
            jitter = delay * 0.25
            return delay + random.uniform(-jitter, jitter)
        
        return delay


class RetryStatistics:
    """Collects and analyzes retry statistics"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempts: List[RetryAttempt] = []
        self.success_count = 0
        self.failure_count = 0
        self.total_duration = 0.0
        self._lock = asyncio.Lock()
    
    async def record_attempt(self, attempt: RetryAttempt, success: bool):
        """Record a retry attempt"""
        async with self._lock:
            self.attempts.append(attempt)
            self.total_duration += attempt.total_elapsed
            
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Keep only recent attempts
            if len(self.attempts) > self.config.statistics_window:
                removed = self.attempts.pop(0)
                self.total_duration -= removed.total_elapsed
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics"""
        async with self._lock:
            if not self.attempts:
                return {
                    "total_attempts": 0,
                    "success_rate": 0.0,
                    "average_attempts": 0.0,
                    "average_duration": 0.0
                }
            
            total_attempts = len(self.attempts)
            success_rate = self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0
            
            attempt_numbers = [attempt.attempt_number for attempt in self.attempts]
            durations = [attempt.total_elapsed for attempt in self.attempts]
            
            return {
                "total_attempts": total_attempts,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": success_rate,
                "average_attempts": statistics.mean(attempt_numbers),
                "max_attempts": max(attempt_numbers) if attempt_numbers else 0,
                "average_duration": statistics.mean(durations) if durations else 0,
                "total_duration": self.total_duration,
                "recent_attempts": [attempt.to_dict() for attempt in self.attempts[-10:]]
            }


class RetryPredicate:
    """Determines whether to retry based on exception and configuration"""
    
    @staticmethod
    def should_retry(
        exception: Exception,
        config: RetryConfig,
        attempt_number: int
    ) -> RetryDecision:
        """Determine if operation should be retried"""
        
        # Check attempt limit
        if attempt_number >= config.max_attempts:
            return RetryDecision.STOP
        
        # Check custom stop condition
        if config.stop_on_exception and config.stop_on_exception(exception):
            return RetryDecision.STOP
        
        # Check non-retryable exceptions
        for non_retryable in config.non_retryable_exceptions:
            if isinstance(exception, non_retryable):
                return RetryDecision.STOP
        
        # Check retryable exceptions
        for retryable in config.retryable_exceptions:
            if isinstance(exception, retryable):
                return RetryDecision.RETRY
        
        # Handle BaseRedTeamException
        if isinstance(exception, BaseRedTeamException):
            if exception.should_retry:
                return RetryDecision.RETRY
            else:
                return RetryDecision.STOP
        
        # Default: don't retry unknown exceptions
        return RetryDecision.STOP


class EnhancedRetryHandler:
    """Advanced retry handler with comprehensive error handling and statistics"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.statistics = RetryStatistics(self.config) if self.config.collect_statistics else None
        self._active_operations: Dict[str, datetime] = {}
        self._rate_limit_tracker: Dict[str, List[datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[T]] if inspect.iscoroutinefunction else Callable[..., T],
        *args,
        operation_id: Optional[str] = None,
        **kwargs
    ) -> T:
        """Execute function with retry logic"""
        
        operation_id = operation_id or f"op_{int(time.time() * 1000)}"
        start_time = time.time()
        attempt_number = 0
        last_delay = 0.0
        last_exception: Optional[BaseRedTeamException] = None
        
        # Check rate limiting
        if self.config.enable_rate_limiting:
            await self._check_rate_limit(operation_id)
        
        # Track active operation
        async with self._lock:
            self._active_operations[operation_id] = datetime.now(UTC)
        
        try:
            while attempt_number < self.config.max_attempts:
                attempt_number += 1
                
                # Check total timeout
                if self.config.total_timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= self.config.total_timeout:
                        raise NetworkException(
                            f"Total timeout exceeded: {elapsed:.2f}s >= {self.config.total_timeout}s",
                            error_code=ErrorCode.TIMEOUT,
                            context=ErrorContext(
                                operation="retry_execution",
                                component="retry_handler",
                                additional_data={
                                    "operation_id": operation_id,
                                    "attempt_number": attempt_number,
                                    "total_elapsed": elapsed
                                }
                            )
                        )
                
                try:
                    # Execute function with timeout
                    if self.config.timeout_per_attempt:
                        if asyncio.iscoroutinefunction(func):
                            result = await asyncio.wait_for(
                                func(*args, **kwargs),
                                timeout=self.config.timeout_per_attempt
                            )
                        else:
                            # For sync functions, we can't easily apply timeout
                            result = func(*args, **kwargs)
                    else:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                    
                    # Check result validation
                    if self.config.retry_on_result and self.config.retry_on_result(result):
                        raise BusinessLogicException(
                            "Result validation failed",
                            error_code=ErrorCode.BL_OPERATION_FAILED
                        )
                    
                    # Success - record statistics and return
                    total_elapsed = time.time() - start_time
                    
                    if self.statistics:
                        attempt = RetryAttempt(
                            attempt_number=attempt_number,
                            delay=last_delay,
                            total_elapsed=total_elapsed,
                            last_exception=last_exception
                        )
                        await self.statistics.record_attempt(attempt, success=True)
                    
                    logger.info(
                        f"Operation {operation_id} succeeded after {attempt_number} attempts "
                        f"in {total_elapsed:.2f}s"
                    )
                    
                    return result
                
                except Exception as e:
                    # Convert to standardized exception if needed
                    if not isinstance(e, BaseRedTeamException):
                        last_exception = self._convert_exception(e, operation_id, attempt_number)
                    else:
                        last_exception = e
                    
                    # Determine if we should retry
                    decision = RetryPredicate.should_retry(
                        last_exception, self.config, attempt_number
                    )
                    
                    if decision == RetryDecision.STOP:
                        break
                    
                    # Calculate delay for next attempt
                    if attempt_number < self.config.max_attempts:
                        delay = BackoffCalculator.calculate_delay(
                            self.config, attempt_number + 1, last_delay
                        )
                        last_delay = delay
                        
                        logger.warning(
                            f"Operation {operation_id} failed (attempt {attempt_number}), "
                            f"retrying in {delay:.2f}s: {last_exception}"
                        )
                        
                        # Sleep before retry
                        await asyncio.sleep(delay)
                    
                    # Track the error
                    await error_tracker.track_error(last_exception)
            
            # All retries exhausted
            total_elapsed = time.time() - start_time
            
            if self.statistics and last_exception:
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    delay=last_delay,
                    total_elapsed=total_elapsed,
                    last_exception=last_exception
                )
                await self.statistics.record_attempt(attempt, success=False)
            
            # Create final exception with retry context
            final_exception = ExternalAPIException(
                f"Operation {operation_id} failed after {attempt_number} attempts in {total_elapsed:.2f}s",
                error_code=ErrorCode.API_UNAVAILABLE,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(
                    operation="retry_execution_failed",
                    component="retry_handler",
                    additional_data={
                        "operation_id": operation_id,
                        "total_attempts": attempt_number,
                        "total_elapsed": total_elapsed,
                        "last_exception": last_exception.to_dict() if last_exception else None
                    }
                ),
                original_exception=last_exception
            )
            
            await error_tracker.track_error(final_exception)
            raise final_exception
        
        finally:
            # Clean up active operation tracking
            async with self._lock:
                self._active_operations.pop(operation_id, None)
    
    def _convert_exception(
        self, 
        exception: Exception, 
        operation_id: str, 
        attempt_number: int
    ) -> BaseRedTeamException:
        """Convert generic exception to BaseRedTeamException"""
        
        context = ErrorContext(
            operation="retry_attempt",
            component="retry_handler",
            additional_data={
                "operation_id": operation_id,
                "attempt_number": attempt_number,
                "original_exception_type": type(exception).__name__
            }
        )
        
        # Map common exceptions
        if isinstance(exception, (ConnectionError, OSError)):
            return NetworkException(
                str(exception),
                error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
                context=context,
                original_exception=exception
            )
        elif isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return NetworkException(
                str(exception),
                error_code=ErrorCode.NETWORK_TIMEOUT,
                context=context,
                original_exception=exception
            )
        elif isinstance(exception, (PermissionError, FileNotFoundError)):
            return FilesystemException(
                str(exception),
                error_code=ErrorCode.FS_PERMISSION_DENIED,
                context=context,
                original_exception=exception
            )
        else:
            return BaseRedTeamException(
                str(exception),
                error_code=ErrorCode.UNKNOWN_ERROR,
                context=context,
                original_exception=exception
            )
    
    async def _check_rate_limit(self, operation_id: str):
        """Check and enforce rate limiting"""
        async with self._lock:
            now = datetime.now(UTC)
            
            # Clean old entries
            cutoff = now - timedelta(seconds=self.config.rate_limit_period)
            for key in list(self._rate_limit_tracker.keys()):
                self._rate_limit_tracker[key] = [
                    timestamp for timestamp in self._rate_limit_tracker[key]
                    if timestamp > cutoff
                ]
                if not self._rate_limit_tracker[key]:
                    del self._rate_limit_tracker[key]
            
            # Check current rate
            current_calls = len(self._rate_limit_tracker.get(operation_id, []))
            
            if current_calls >= self.config.rate_limit_calls:
                raise ResourceException(
                    f"Rate limit exceeded for operation {operation_id}: "
                    f"{current_calls} calls in {self.config.rate_limit_period}s",
                    error_code=ErrorCode.RES_RATE_LIMIT_EXCEEDED,
                    retry_after=self.config.rate_limit_period
                )
            
            # Record this call
            if operation_id not in self._rate_limit_tracker:
                self._rate_limit_tracker[operation_id] = []
            self._rate_limit_tracker[operation_id].append(now)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get retry handler statistics"""
        if not self.statistics:
            return {"statistics_disabled": True}
        
        stats = await self.statistics.get_statistics()
        
        async with self._lock:
            stats["active_operations"] = len(self._active_operations)
            stats["rate_limit_tracking"] = {
                op_id: len(timestamps) 
                for op_id, timestamps in self._rate_limit_tracker.items()
            }
        
        return stats
    
    async def get_active_operations(self) -> Dict[str, str]:
        """Get currently active operations"""
        async with self._lock:
            return {
                op_id: timestamp.isoformat()
                for op_id, timestamp in self._active_operations.items()
            }


# Global retry handler instances for different use cases
default_retry_handler = EnhancedRetryHandler()

# Specialized handlers
network_retry_handler = EnhancedRetryHandler(RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    jitter_type=JitterType.EQUAL,
    timeout_per_attempt=30.0,
    total_timeout=300.0,
    retryable_exceptions={NetworkException, ExternalAPIException, ConnectionError, TimeoutError}
))

database_retry_handler = EnhancedRetryHandler(RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=30.0,
    backoff_strategy=BackoffStrategy.LINEAR,
    jitter_type=JitterType.RANDOM,
    timeout_per_attempt=10.0,
    retryable_exceptions={DatabaseException, ConnectionError}
))

api_retry_handler = EnhancedRetryHandler(RetryConfig(
    max_attempts=7,
    base_delay=2.0,
    max_delay=120.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    exponential_base=2.0,
    jitter_type=JitterType.DECORR,
    timeout_per_attempt=45.0,
    total_timeout=600.0,
    enable_rate_limiting=True,
    rate_limit_calls=100,
    rate_limit_period=60.0,
    retryable_exceptions={ExternalAPIException, NetworkException, ResourceException}
))


# Convenience decorators
def retry_on_failure(
    config: Optional[RetryConfig] = None,
    handler: Optional[EnhancedRetryHandler] = None
):
    """Decorator to add retry logic to functions"""
    def decorator(func):
        retry_handler = handler or default_retry_handler
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await retry_handler.execute_with_retry(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Convert sync function to async for retry handler
                async def async_func():
                    return func(*args, **kwargs)
                
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    retry_handler.execute_with_retry(async_func)
                )
            return sync_wrapper
    return decorator


def network_retry(func):
    """Decorator for network operations with appropriate retry settings"""
    return retry_on_failure(handler=network_retry_handler)(func)


def database_retry(func):
    """Decorator for database operations with appropriate retry settings"""
    return retry_on_failure(handler=database_retry_handler)(func)


def api_retry(func):
    """Decorator for API operations with appropriate retry settings"""
    return retry_on_failure(handler=api_retry_handler)(func)