# ABOUTME: Circuit breaker pattern implementation for robust external API handling
# ABOUTME: Provides automatic failure detection, fallback mechanisms, and self-healing

import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
import statistics
from abc import ABC, abstractmethod

from .exceptions import (
    BaseRedTeamException, NetworkException, ExternalAPIException, 
    ErrorCode, ErrorSeverity, ErrorCategory, ErrorContext, error_tracker
)

T = TypeVar('T')

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5              # Number of failures to open circuit
    recovery_timeout: float = 60.0          # Seconds to wait before trying to recover
    success_threshold: int = 3              # Consecutive successes needed to close circuit
    timeout: float = 30.0                   # Default timeout for operations
    slow_call_threshold: float = 10.0       # Calls slower than this are considered failures
    minimum_calls: int = 10                 # Minimum calls before evaluating failure rate
    failure_rate_threshold: float = 0.5     # Failure rate (0.0-1.0) to open circuit
    evaluation_window: float = 60.0         # Time window for failure rate calculation
    exponential_backoff: bool = True        # Use exponential backoff for recovery attempts
    max_recovery_timeout: float = 300.0     # Maximum recovery timeout with backoff


@dataclass
class CallResult:
    """Result of a circuit breaker call"""
    success: bool
    duration: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: Optional[BaseRedTeamException] = None
    
    @property
    def is_slow(self) -> bool:
        """Check if call was considered slow"""
        return self.duration > 10.0  # Configurable threshold


class CircuitBreakerMetrics:
    """Metrics collection for circuit breaker"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.call_history: list[CallResult] = []
        self.state_history: list[tuple[CircuitState, datetime]] = []
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self._lock = asyncio.Lock()
    
    async def record_call(self, result: CallResult):
        """Record a call result"""
        async with self._lock:
            self.call_history.append(result)
            self.total_calls += 1
            
            if result.success:
                self.total_successes += 1
                self.consecutive_successes += 1
                self.consecutive_failures = 0
            else:
                self.total_failures += 1
                self.consecutive_failures += 1
                self.consecutive_successes = 0
            
            # Keep history within evaluation window
            cutoff_time = datetime.now(UTC) - timedelta(seconds=self.config.evaluation_window)
            self.call_history = [
                call for call in self.call_history 
                if call.timestamp > cutoff_time
            ]
    
    async def record_state_change(self, new_state: CircuitState):
        """Record a state change"""
        async with self._lock:
            self.state_history.append((new_state, datetime.now(UTC)))
            
            # Keep last 100 state changes
            if len(self.state_history) > 100:
                self.state_history = self.state_history[-100:]
    
    async def get_failure_rate(self) -> float:
        """Calculate current failure rate"""
        async with self._lock:
            if len(self.call_history) < self.config.minimum_calls:
                return 0.0
            
            recent_calls = [
                call for call in self.call_history
                if call.timestamp > datetime.now(UTC) - timedelta(seconds=self.config.evaluation_window)
            ]
            
            if not recent_calls:
                return 0.0
            
            failures = sum(1 for call in recent_calls if not call.success or call.is_slow)
            return failures / len(recent_calls)
    
    async def get_average_response_time(self) -> float:
        """Get average response time for recent calls"""
        async with self._lock:
            recent_calls = [
                call for call in self.call_history
                if call.timestamp > datetime.now(UTC) - timedelta(seconds=self.config.evaluation_window)
            ]
            
            if not recent_calls:
                return 0.0
            
            return statistics.mean(call.duration for call in recent_calls)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        async with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "consecutive_failures": self.consecutive_failures,
                "consecutive_successes": self.consecutive_successes,
                "failure_rate": await self.get_failure_rate(),
                "average_response_time": await self.get_average_response_time(),
                "recent_calls_count": len(self.call_history),
                "state_changes": len(self.state_history)
            }


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies"""
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute fallback logic"""
        pass


class CachedFallback(FallbackStrategy):
    """Fallback using cached responses"""
    
    def __init__(self, cache: Dict[str, Any], cache_key_generator: Callable = None):
        self.cache = cache
        self.cache_key_generator = cache_key_generator or self._default_key_generator
    
    def _default_key_generator(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        return f"{args}:{kwargs}"
    
    async def execute(self, *args, **kwargs) -> Any:
        """Return cached response if available"""
        cache_key = self.cache_key_generator(*args, **kwargs)
        
        if cache_key in self.cache:
            logger.info(f"Using cached fallback for key: {cache_key}")
            return self.cache[cache_key]
        
        raise ExternalAPIException(
            "No cached fallback available",
            error_code=ErrorCode.API_UNAVAILABLE,
            context=ErrorContext(
                operation="fallback_execution",
                component="circuit_breaker",
                additional_data={"cache_key": cache_key}
            )
        )


class DefaultValueFallback(FallbackStrategy):
    """Fallback using default values"""
    
    def __init__(self, default_value: Any):
        self.default_value = default_value
    
    async def execute(self, *args, **kwargs) -> Any:
        """Return default value"""
        logger.info(f"Using default value fallback: {self.default_value}")
        return self.default_value


class AlternativeServiceFallback(FallbackStrategy):
    """Fallback using alternative service"""
    
    def __init__(self, alternative_function: Callable):
        self.alternative_function = alternative_function
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute alternative service"""
        logger.info("Using alternative service fallback")
        try:
            if asyncio.iscoroutinefunction(self.alternative_function):
                return await self.alternative_function(*args, **kwargs)
            else:
                return self.alternative_function(*args, **kwargs)
        except Exception as e:
            raise ExternalAPIException(
                f"Alternative service also failed: {str(e)}",
                error_code=ErrorCode.API_UNAVAILABLE,
                original_exception=e
            )


class CircuitBreaker:
    """Circuit breaker implementation for robust external service calls"""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_strategy: Optional[FallbackStrategy] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_strategy = fallback_strategy
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None
        self.metrics = CircuitBreakerMetrics(self.config)
        self._lock = asyncio.Lock()
        self._recovery_attempts = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            current_state = await self._get_current_state()
            
            if current_state == CircuitState.OPEN:
                return await self._handle_open_circuit(func, *args, **kwargs)
            elif current_state == CircuitState.HALF_OPEN:
                return await self._handle_half_open_circuit(func, *args, **kwargs)
            else:  # CLOSED
                return await self._handle_closed_circuit(func, *args, **kwargs)
    
    async def _get_current_state(self) -> CircuitState:
        """Determine current circuit state"""
        if self.state == CircuitState.OPEN:
            if await self._should_attempt_recovery():
                await self._transition_to_half_open()
                return CircuitState.HALF_OPEN
            return CircuitState.OPEN
        
        elif self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                await self._transition_to_closed()
                return CircuitState.CLOSED
            elif self.metrics.consecutive_failures > 0:
                await self._transition_to_open()
                return CircuitState.OPEN
            return CircuitState.HALF_OPEN
        
        else:  # CLOSED
            if await self._should_open_circuit():
                await self._transition_to_open()
                return CircuitState.OPEN
            return CircuitState.CLOSED
    
    async def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery from open state"""
        if not self.last_failure_time:
            return True
        
        timeout = self.config.recovery_timeout
        if self.config.exponential_backoff:
            timeout = min(
                timeout * (2 ** self._recovery_attempts),
                self.config.max_recovery_timeout
            )
        
        time_since_failure = (datetime.now(UTC) - self.last_failure_time).total_seconds()
        return time_since_failure >= timeout
    
    async def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        # Check consecutive failures threshold
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        failure_rate = await self.metrics.get_failure_rate()
        return (
            len(self.metrics.call_history) >= self.config.minimum_calls and
            failure_rate >= self.config.failure_rate_threshold
        )
    
    async def _transition_to_open(self):
        """Transition circuit to open state"""
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now(UTC)
        self._recovery_attempts += 1
        
        await self.metrics.record_state_change(CircuitState.OPEN)
        
        logger.warning(
            f"Circuit breaker '{self.name}' opened due to failures. "
            f"Recovery attempt #{self._recovery_attempts}"
        )
        
        # Track the circuit opening as an error
        error = ExternalAPIException(
            f"Circuit breaker '{self.name}' opened due to excessive failures",
            error_code=ErrorCode.API_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                operation="circuit_state_change",
                component="circuit_breaker",
                additional_data={
                    "circuit_name": self.name,
                    "new_state": "open",
                    "recovery_attempts": self._recovery_attempts,
                    "consecutive_failures": self.metrics.consecutive_failures
                }
            )
        )
        await error_tracker.track_error(error)
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        self.state = CircuitState.HALF_OPEN
        await self.metrics.record_state_change(CircuitState.HALF_OPEN)
        
        logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state"""
        self.state = CircuitState.CLOSED
        self._recovery_attempts = 0
        await self.metrics.record_state_change(CircuitState.CLOSED)
        
        logger.info(f"Circuit breaker '{self.name}' recovered and closed")
    
    async def _handle_open_circuit(self, func: Callable, *args, **kwargs) -> T:
        """Handle call when circuit is open"""
        if self.fallback_strategy:
            try:
                result = await self.fallback_strategy.execute(*args, **kwargs)
                logger.info(f"Circuit breaker '{self.name}' used fallback successfully")
                return result
            except Exception as e:
                logger.error(f"Circuit breaker '{self.name}' fallback failed: {e}")
        
        # No fallback or fallback failed
        raise ExternalAPIException(
            f"Circuit breaker '{self.name}' is open - service unavailable",
            error_code=ErrorCode.API_UNAVAILABLE,
            severity=ErrorSeverity.MEDIUM,
            retry_after=self.config.recovery_timeout,
            context=ErrorContext(
                operation="circuit_breaker_call",
                component="circuit_breaker",
                additional_data={
                    "circuit_name": self.name,
                    "circuit_state": "open",
                    "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
                }
            )
        )
    
    async def _handle_half_open_circuit(self, func: Callable, *args, **kwargs) -> T:
        """Handle call when circuit is half-open"""
        return await self._execute_function(func, *args, **kwargs)
    
    async def _handle_closed_circuit(self, func: Callable, *args, **kwargs) -> T:
        """Handle call when circuit is closed"""
        return await self._execute_function(func, *args, **kwargs)
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> T:
        """Execute the actual function with timeout and error handling"""
        start_time = time.time()
        
        try:
            # Apply timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Record successful call
            await self.metrics.record_call(CallResult(
                success=True,
                duration=duration
            ))
            
            return result
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            
            error = NetworkException(
                f"Function call timed out after {self.config.timeout}s",
                error_code=ErrorCode.NETWORK_TIMEOUT,
                context=ErrorContext(
                    operation="circuit_breaker_call",
                    component="circuit_breaker",
                    additional_data={
                        "circuit_name": self.name,
                        "timeout": self.config.timeout,
                        "duration": duration
                    }
                )
            )
            
            # Record failed call
            await self.metrics.record_call(CallResult(
                success=False,
                duration=duration,
                error=error
            ))
            
            await error_tracker.track_error(error)
            raise error
            
        except BaseRedTeamException as e:
            duration = time.time() - start_time
            
            # Record failed call
            await self.metrics.record_call(CallResult(
                success=False,
                duration=duration,
                error=e
            ))
            
            raise e
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Convert to standardized exception
            error = ExternalAPIException(
                f"Function call failed: {str(e)}",
                error_code=ErrorCode.API_UNAVAILABLE,
                context=ErrorContext(
                    operation="circuit_breaker_call",
                    component="circuit_breaker",
                    additional_data={
                        "circuit_name": self.name,
                        "original_error": str(e),
                        "duration": duration
                    }
                ),
                original_exception=e
            )
            
            # Record failed call
            await self.metrics.record_call(CallResult(
                success=False,
                duration=duration,
                error=error
            ))
            
            await error_tracker.track_error(error)
            raise error
    
    async def get_state(self) -> CircuitState:
        """Get current circuit state"""
        async with self._lock:
            return await self._get_current_state()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        stats = await self.metrics.get_stats()
        return {
            "name": self.name,
            "state": self.state.value,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recovery_attempts": self._recovery_attempts,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            **stats
        }
    
    async def reset(self):
        """Reset circuit breaker to closed state"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.last_failure_time = None
            self._recovery_attempts = 0
            await self.metrics.record_state_change(CircuitState.CLOSED)
            
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    async def force_open(self):
        """Force circuit breaker to open state"""
        async with self._lock:
            await self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' manually opened")


# Global circuit breaker registry
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_strategy: Optional[FallbackStrategy] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config, fallback_strategy)
                logger.info(f"Created new circuit breaker: {name}")
            
            return self.breakers[name]
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        async with self._lock:
            metrics = {}
            for name, breaker in self.breakers.items():
                metrics[name] = await breaker.get_metrics()
            return metrics
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        async with self._lock:
            for breaker in self.breakers.values():
                await breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


# Convenience decorator for circuit breaker protection
def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback_strategy: Optional[FallbackStrategy] = None
):
    """Decorator to apply circuit breaker protection to functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            breaker = await circuit_registry.get_or_create(name, config, fallback_strategy)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator