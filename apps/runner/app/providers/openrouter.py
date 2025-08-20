# ABOUTME: Enhanced OpenRouter async client with comprehensive error handling, circuit breakers, and monitoring
# ABOUTME: Handles API calls, retries on failures, tracks costs, and provides robust error recovery

import os
import httpx
import asyncio
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_combine,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta

from ..util.exceptions import (
    BaseRedTeamException, NetworkException, ExternalAPIException, 
    AuthenticationException, ResourceException, ValidationException,
    ErrorCode, ErrorSeverity, ErrorContext, error_tracker, handle_exceptions
)
from ..util.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CachedFallback, 
    DefaultValueFallback, circuit_registry
)
from ..util.retry import (
    EnhancedRetryHandler, RetryConfig, BackoffStrategy, JitterType,
    api_retry_handler
)
from ..util.validation import validate_input, global_validator
from ..monitoring.health import add_custom_metric, MetricType

logger = logging.getLogger(__name__)

# Enhanced exception classes for OpenRouter-specific errors
class OpenRouterException(ExternalAPIException):
    """Base exception for OpenRouter API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class OpenRouterRateLimitError(ResourceException):
    """Raised when rate limited by OpenRouter API"""
    def __init__(self, message: str, retry_after: int = 5, **kwargs):
        super().__init__(
            message, 
            error_code=ErrorCode.RES_RATE_LIMIT_EXCEEDED,
            retry_after=retry_after,
            should_retry=True,
            **kwargs
        )
        self.retry_after = retry_after


class OpenRouterAuthError(AuthenticationException):
    """Raised when authentication fails with OpenRouter API"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.AUTH_INVALID_KEY,
            should_retry=False,
            **kwargs
        )


class OpenRouterQuotaError(ResourceException):
    """Raised when quota is exceeded"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.RES_QUOTA_EXCEEDED,
            should_retry=False,
            **kwargs
        )


class OpenRouterTimeoutError(NetworkException):
    """Raised when request times out"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.NETWORK_TIMEOUT,
            should_retry=True,
            **kwargs
        )

# Enhanced wait strategy for OpenRouter with respect for retry_after headers
class OpenRouterWaitStrategy:
    """Custom wait strategy that respects OpenRouter API rate limit headers"""
    
    def __init__(self, exponential_base=2, exponential_max=120):
        self.exponential = wait_exponential(multiplier=exponential_base, min=1, max=exponential_max)
    
    def __call__(self, retry_state):
        # Check if the last exception was a rate limit error with retry_after
        if retry_state.outcome and retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if isinstance(exception, OpenRouterRateLimitError) and hasattr(exception, 'retry_after'):
                wait_time = exception.retry_after
                logger.info(f"OpenRouter rate limit: waiting {wait_time}s as instructed by API")
                return wait_time
        
        # Fall back to exponential backoff for other errors
        return self.exponential(retry_state)


# Enhanced retry logic for OpenRouter specifics
def should_retry_openrouter(exception):
    """Custom retry logic for OpenRouter-specific exceptions"""
    if isinstance(exception, OpenRouterRateLimitError):
        return True
    if isinstance(exception, OpenRouterTimeoutError):
        return True
    if isinstance(exception, OpenRouterException):
        # Don't retry auth errors or quota errors
        if isinstance(exception, (OpenRouterAuthError, OpenRouterQuotaError)):
            return False
        # Retry server errors (5xx)
        if exception.status_code and exception.status_code >= 500:
            return True
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on specific status codes
        return exception.response.status_code in [429, 500, 502, 503, 504, 520, 522, 524]
    if isinstance(exception, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    return False


# Enhanced retry configuration
OPENROUTER_RETRY_CONFIG = RetryConfig(
    max_attempts=8,  # Increased for better resilience
    base_delay=1.0,
    max_delay=300.0,  # 5 minutes max
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    exponential_base=2.0,
    jitter_type=JitterType.EQUAL,
    jitter_amount=0.2,
    timeout_per_attempt=120.0,  # 2 minutes per attempt
    total_timeout=900.0,  # 15 minutes total
    retryable_exceptions={
        OpenRouterRateLimitError, OpenRouterTimeoutError, OpenRouterException,
        NetworkException, ExternalAPIException, httpx.HTTPStatusError,
        httpx.TimeoutException, httpx.ConnectError, httpx.ReadError
    },
    non_retryable_exceptions={
        OpenRouterAuthError, OpenRouterQuotaError, ValidationException
    },
    stop_on_exception=lambda e: (
        isinstance(e, OpenRouterException) and 
        isinstance(e, (OpenRouterAuthError, OpenRouterQuotaError))
    )
)

# Enhanced model pricing with more comprehensive coverage
MODEL_PRICING = {
    # OpenAI models
    "openai/gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
    "openai/gpt-4": {"prompt": 30.0, "completion": 60.0},
    "openai/gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "openai/gpt-4o": {"prompt": 5.0, "completion": 15.0},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "openai/o1-preview": {"prompt": 15.0, "completion": 60.0},
    "openai/o1-mini": {"prompt": 3.0, "completion": 12.0},
    
    # Competition target model
    "openai/gpt-oss-20b": {"prompt": 2.0, "completion": 6.0},  # Placeholder for competition
    
    # Meta Llama models
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.07, "completion": 0.07},
    "meta-llama/llama-3.1-70b-instruct": {"prompt": 0.59, "completion": 0.79},
    "meta-llama/llama-3.1-405b-instruct": {"prompt": 2.7, "completion": 2.7},
    "meta-llama/llama-3-8b-instruct": {"prompt": 0.07, "completion": 0.07},
    "meta-llama/llama-3-70b-instruct": {"prompt": 0.59, "completion": 0.79},
    "meta-llama/llama-2-7b-chat": {"prompt": 0.07, "completion": 0.07},
    "meta-llama/llama-2-13b-chat": {"prompt": 0.13, "completion": 0.13},
    "meta-llama/llama-2-70b-chat": {"prompt": 0.59, "completion": 0.79},
    
    # Anthropic models
    "anthropic/claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    "anthropic/claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
    "anthropic/claude-3-opus": {"prompt": 15.0, "completion": 75.0},
    "anthropic/claude-3.5-sonnet": {"prompt": 3.0, "completion": 15.0},
    
    # Google models
    "google/gemini-pro": {"prompt": 0.5, "completion": 1.5},
    "google/gemini-pro-vision": {"prompt": 0.5, "completion": 1.5},
    "google/gemma-7b-it": {"prompt": 0.07, "completion": 0.07},
    "google/gemma-2-9b-it": {"prompt": 0.09, "completion": 0.09},
    "google/gemma-2-27b-it": {"prompt": 0.27, "completion": 0.27},
    
    # Mistral models
    "mistralai/mistral-7b-instruct": {"prompt": 0.07, "completion": 0.07},
    "mistralai/mixtral-8x7b-instruct": {"prompt": 0.24, "completion": 0.24},
    "mistralai/mixtral-8x22b-instruct": {"prompt": 0.65, "completion": 0.65},
    
    # Default fallback pricing
    "default": {"prompt": 1.0, "completion": 2.0}
}


@dataclass
class RequestMetrics:
    """Metrics for tracking API request performance"""
    request_id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "unknown"  # success, error, timeout, rate_limited
    error_type: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "error_type": self.error_type,
            "retry_count": self.retry_count
        }


@dataclass
class EnhancedOpenRouterClient:
    """Enhanced async client for OpenRouter API with comprehensive error handling and monitoring"""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    timeout: int = 120  # Increased default timeout
    total_cost: float = 0.0
    total_tokens: int = 0
    request_count: int = 0
    error_count: int = 0
    
    # Private fields
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _circuit_breaker: Optional[CircuitBreaker] = field(default=None, init=False)
    _retry_handler: Optional[EnhancedRetryHandler] = field(default=None, init=False)
    _request_metrics: List[RequestMetrics] = field(default_factory=list, init=False)
    _metrics_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _last_reset_time: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    
    # Configuration
    enable_circuit_breaker: bool = True
    enable_enhanced_retry: bool = True
    enable_metrics_collection: bool = True
    max_metrics_history: int = 1000
    cost_warning_threshold: float = 100.0  # USD
    rate_limit_buffer: float = 0.1  # 10% buffer for rate limits

    def __post_init__(self):
        # Validate API key
        if not self.api_key:
            raise OpenRouterAuthError(
                "OPENROUTER_API_KEY environment variable not set",
                context=ErrorContext(
                    operation="client_initialization",
                    component="openrouter_client"
                )
            )
        
        # Validate API key format
        validation_result = global_validator.validate_field("api_key", self.api_key)
        if not validation_result.is_valid:
            raise OpenRouterAuthError(
                f"Invalid API key format: {'; '.join(validation_result.errors)}",
                context=ErrorContext(
                    operation="api_key_validation",
                    component="openrouter_client"
                )
            )
        
        # Initialize circuit breaker
        if self.enable_circuit_breaker:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3,
                timeout=self.timeout,
                minimum_calls=10,
                failure_rate_threshold=0.5,
                evaluation_window=300.0  # 5 minutes
            )
            
            # Create fallback strategy with cached responses
            cache = {}  # In production, this would be a proper cache
            fallback = CachedFallback(cache)
            
            asyncio.create_task(self._initialize_circuit_breaker(circuit_config, fallback))
        
        # Initialize enhanced retry handler
        if self.enable_enhanced_retry:
            self._retry_handler = EnhancedRetryHandler(OPENROUTER_RETRY_CONFIG)
        
        logger.info(f"Initialized EnhancedOpenRouterClient with circuit_breaker={self.enable_circuit_breaker}, enhanced_retry={self.enable_enhanced_retry}")
    
    async def _initialize_circuit_breaker(self, config: CircuitBreakerConfig, fallback):
        """Initialize circuit breaker asynchronously"""
        self._circuit_breaker = await circuit_registry.get_or_create(
            "openrouter_api",
            config,
            fallback
        )
    
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with comprehensive configuration"""
        async with self._client_lock:
            if self._client is None:
                # Enhanced client configuration for robustness
                limits = httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=30.0
                )
                
                # Comprehensive timeout configuration
                timeouts = httpx.Timeout(
                    connect=10.0,      # Connection timeout
                    read=self.timeout,  # Read timeout
                    write=10.0,        # Write timeout
                    pool=5.0           # Pool timeout
                )
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/harshitIIITD/harshit-kaggle-red-teamining",
                    "X-Title": "Red-Teaming System",
                    "User-Agent": "RedTeamClient/1.0"
                }
                
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=timeouts,
                    limits=limits,
                    follow_redirects=True,
                    verify=True  # Ensure SSL verification
                )
                
                logger.info("Created new HTTP client for OpenRouter API")
            
            return self._client
    
    async def _close_client(self):
        """Close the HTTP client safely"""
        async with self._client_lock:
            if self._client:
                await self._client.aclose()
                self._client = None
                logger.info("Closed HTTP client")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self._close_client()
    
    @handle_exceptions(
        component="openrouter_client",
        operation="request_validation"
    )
    async def _validate_request(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **params
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """Validate and sanitize request parameters"""
        
        # Validate model name
        model_validation = global_validator.validate_field("model_name", model)
        if not model_validation.is_valid:
            raise ValidationException(
                f"Invalid model name: {'; '.join(model_validation.errors)}",
                error_code=ErrorCode.VAL_INVALID_INPUT,
                context=ErrorContext(
                    operation="model_validation",
                    component="openrouter_client",
                    additional_data={"model": model}
                )
            )
        
        # Use sanitized model name
        if model_validation.sanitized_data:
            model = model_validation.sanitized_data
        
        # Validate messages structure
        if not messages or not isinstance(messages, list):
            raise ValidationException(
                "Messages must be a non-empty list",
                error_code=ErrorCode.VAL_INVALID_INPUT
            )
        
        sanitized_messages = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationException(
                    f"Message {i} must be a dictionary",
                    error_code=ErrorCode.VAL_INVALID_INPUT
                )
            
            # Validate required fields
            if "role" not in message or "content" not in message:
                raise ValidationException(
                    f"Message {i} must have 'role' and 'content' fields",
                    error_code=ErrorCode.VAL_MISSING_REQUIRED
                )
            
            # Validate role
            valid_roles = {"system", "user", "assistant", "function"}
            if message["role"] not in valid_roles:
                raise ValidationException(
                    f"Invalid role '{message['role']}' in message {i}",
                    error_code=ErrorCode.VAL_INVALID_INPUT
                )
            
            # Validate and sanitize content
            content_validation = global_validator.validate_field("user_input", message["content"])
            if not content_validation.is_valid:
                # Log security violations but don't block (this is red-teaming)
                logger.warning(f"Message content validation warnings: {content_validation.errors}")
            
            sanitized_message = {
                "role": message["role"],
                "content": content_validation.sanitized_data or message["content"]
            }
            
            # Preserve other fields if present
            for key, value in message.items():
                if key not in {"role", "content"}:
                    sanitized_message[key] = value
            
            sanitized_messages.append(sanitized_message)
        
        # Validate additional parameters
        sanitized_params = {}
        for key, value in params.items():
            if key in {"temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"}:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValidationException(
                        f"Parameter '{key}' must be a non-negative number",
                        error_code=ErrorCode.VAL_RANGE_ERROR
                    )
                sanitized_params[key] = float(value)
            
            elif key in {"max_tokens", "n"}:
                if not isinstance(value, int) or value <= 0:
                    raise ValidationException(
                        f"Parameter '{key}' must be a positive integer",
                        error_code=ErrorCode.VAL_RANGE_ERROR
                    )
                sanitized_params[key] = value
            
            elif key in {"stop"}:
                if isinstance(value, str):
                    sanitized_params[key] = [value]
                elif isinstance(value, list):
                    sanitized_params[key] = [str(item) for item in value]
                else:
                    raise ValidationException(
                        f"Parameter '{key}' must be a string or list of strings",
                        error_code=ErrorCode.VAL_INVALID_INPUT
                    )
            
            else:
                # Pass through other parameters with basic validation
                if isinstance(value, str):
                    content_validation = global_validator.validate_field("user_input", value)
                    sanitized_params[key] = content_validation.sanitized_data or value
                else:
                    sanitized_params[key] = value
        
        return model, sanitized_messages, sanitized_params
    
    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage with enhanced error handling"""
        try:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Get pricing, fallback to default if model not found
            pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
            
            prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
            completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
            
            total_cost = prompt_cost + completion_cost
            
            # Log cost if it's significant
            if total_cost > 0.01:  # More than 1 cent
                logger.info(f"API call cost: ${total_cost:.4f} for model {model} "
                           f"({prompt_tokens} prompt + {completion_tokens} completion tokens)")
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating cost for model {model}: {e}")
            return 0.0
    
    async def _record_metrics(self, metrics: RequestMetrics):
        """Record request metrics with thread safety"""
        if not self.enable_metrics_collection:
            return
        
        async with self._metrics_lock:
            self._request_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self._request_metrics) > self.max_metrics_history:
                self._request_metrics = self._request_metrics[-self.max_metrics_history:]
            
            # Update counters
            self.request_count += 1
            if metrics.status == "error":
                self.error_count += 1
            
            self.total_cost += metrics.cost_usd
            self.total_tokens += metrics.total_tokens
        
        # Send metrics to monitoring system
        await add_custom_metric(
            name="openrouter.request.duration_ms",
            value=metrics.duration_ms,
            metric_type=MetricType.TIMER,
            tags={
                "model": metrics.model,
                "status": metrics.status,
                "component": "openrouter_client"
            }
        )
        
        await add_custom_metric(
            name="openrouter.request.cost_usd",
            value=metrics.cost_usd,
            metric_type=MetricType.GAUGE,
            tags={
                "model": metrics.model,
                "component": "openrouter_client"
            }
        )
        
        await add_custom_metric(
            name="openrouter.request.tokens",
            value=metrics.total_tokens,
            metric_type=MetricType.COUNTER,
            tags={
                "model": metrics.model,
                "component": "openrouter_client"
            }
        )
        
        # Check cost warning threshold
        if self.total_cost > self.cost_warning_threshold:
            logger.warning(f"OpenRouter API costs have exceeded ${self.cost_warning_threshold:.2f}. "
                          f"Current total: ${self.total_cost:.2f}")
    
    def _parse_openrouter_error(self, response: httpx.Response) -> OpenRouterException:
        """Parse OpenRouter API error response into appropriate exception"""
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            error_code = error_data.get("error", {}).get("code", "unknown")
            error_type = error_data.get("error", {}).get("type", "unknown")
        except Exception:
            error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            error_code = "parse_error"
            error_type = "unknown"
        
        context = ErrorContext(
            operation="openrouter_api_call",
            component="openrouter_client",
            additional_data={
                "status_code": response.status_code,
                "error_code": error_code,
                "error_type": error_type,
                "url": str(response.url),
                "headers": dict(response.headers)
            }
        )
        
        # Map status codes to appropriate exceptions
        if response.status_code == 401:
            return OpenRouterAuthError(
                f"Authentication failed: {error_message}",
                context=context
            )
        elif response.status_code == 403:
            return OpenRouterQuotaError(
                f"Quota exceeded or forbidden: {error_message}",
                context=context
            )
        elif response.status_code == 429:
            # Extract retry-after header if present
            retry_after = response.headers.get("retry-after", "60")
            try:
                retry_after_seconds = int(retry_after)
            except (ValueError, TypeError):
                retry_after_seconds = 60
            
            return OpenRouterRateLimitError(
                f"Rate limit exceeded: {error_message}",
                retry_after=retry_after_seconds,
                context=context
            )
        elif response.status_code >= 500:
            return OpenRouterException(
                f"Server error: {error_message}",
                status_code=response.status_code,
                error_code=ErrorCode.API_UNAVAILABLE,
                should_retry=True,
                context=context
            )
        else:
            return OpenRouterException(
                f"API error: {error_message}",
                status_code=response.status_code,
                error_code=ErrorCode.API_INVALID_RESPONSE,
                context=context
    
    @handle_exceptions(
        component="openrouter_client",
        operation="chat_completion"
    )
    async def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **params
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced chat completion with comprehensive error handling and monitoring
        
        Args:
            model: Model name to use
            messages: List of message dictionaries with 'role' and 'content'
            **params: Additional parameters for the API call
            
        Returns:
            Tuple of (response_text, metadata)
            
        Raises:
            OpenRouterException: For API-specific errors
            ValidationException: For input validation errors
            NetworkException: For network-related errors
        """
        
        request_id = f"req_{int(time.time() * 1000)}_{id(self)}"
        start_time = time.time()
        retry_count = 0
        
        logger.info(f"Starting chat completion request {request_id} for model {model}")
        
        # Initialize metrics
        metrics = RequestMetrics(
            request_id=request_id,
            model=model
        )
        
        try:
            # Validate and sanitize inputs
            model, messages, params = await self._validate_request(model, messages, **params)
            
            # Prepare the actual API call function
            async def _make_api_call():
                nonlocal retry_count
                retry_count += 1
                
                client = await self._get_client()
                
                payload = {
                    "model": model,
                    "messages": messages,
                    **params
                }
                
                # Add request metadata
                if "metadata" not in payload:
                    payload["metadata"] = {}
                payload["metadata"].update({
                    "request_id": request_id,
                    "client_version": "enhanced_v1.0",
                    "retry_count": retry_count
                })
                
                logger.debug(f"Making API call {request_id} (attempt {retry_count}) to {model}")
                
                # Make the HTTP request
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                # Handle HTTP errors
                if not response.is_success:
                    raise self._parse_openrouter_error(response)
                
                return response.json()
            
            # Execute with circuit breaker and retry logic
            if self.enable_circuit_breaker and self._circuit_breaker:
                if self.enable_enhanced_retry and self._retry_handler:
                    # Use both circuit breaker and enhanced retry
                    response_data = await self._circuit_breaker.call(
                        self._retry_handler.execute_with_retry,
                        _make_api_call,
                        operation_id=request_id
                    )
                else:
                    # Use only circuit breaker
                    response_data = await self._circuit_breaker.call(_make_api_call)
            elif self.enable_enhanced_retry and self._retry_handler:
                # Use only enhanced retry
                response_data = await self._retry_handler.execute_with_retry(
                    _make_api_call,
                    operation_id=request_id
                )
            else:
                # Direct call without protection (not recommended)
                logger.warning("Making unprotected API call - circuit breaker and retry disabled")
                response_data = await _make_api_call()
            
            # Process successful response
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract response content
            if "choices" not in response_data or not response_data["choices"]:
                raise OpenRouterException(
                    "No choices in API response",
                    error_code=ErrorCode.API_INVALID_RESPONSE,
                    context=ErrorContext(
                        operation="response_parsing",
                        component="openrouter_client",
                        additional_data={"response_keys": list(response_data.keys())}
                    )
                )
            
            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                raise OpenRouterException(
                    "Invalid message structure in API response",
                    error_code=ErrorCode.API_INVALID_RESPONSE
                )
            
            content = choice["message"]["content"]
            
            # Extract usage and calculate cost
            usage = response_data.get("usage", {})
            cost = self._calculate_cost(model, usage)
            
            # Update metrics
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.total_tokens = usage.get("total_tokens", 0)
            metrics.duration_ms = duration_ms
            metrics.cost_usd = cost
            metrics.status = "success"
            metrics.retry_count = retry_count
            
            # Record metrics
            await self._record_metrics(metrics)
            
            # Prepare metadata
            metadata = {
                "request_id": request_id,
                "model": model,
                "usage": usage,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "retry_count": retry_count,
                "timestamp": datetime.now(UTC).isoformat(),
                "finish_reason": choice.get("finish_reason"),
                "provider_data": response_data.get("provider", {}),
                "raw_response": response_data  # Include full response for debugging
            }
            
            logger.info(f"Completed chat request {request_id} successfully in {duration_ms:.1f}ms "
                       f"(${cost:.4f}, {usage.get('total_tokens', 0)} tokens, {retry_count} attempts)")
            
            return content, metadata
            
        except OpenRouterException:
            # Already properly formatted exception, just re-raise
            raise
            
        except Exception as e:
            # Handle unexpected errors
            duration_ms = (time.time() - start_time) * 1000
            
            # Update metrics for error
            metrics.duration_ms = duration_ms
            metrics.status = "error"
            metrics.error_type = type(e).__name__
            metrics.retry_count = retry_count
            
            await self._record_metrics(metrics)
            
            # Convert to appropriate exception type
            if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError)):
                error = OpenRouterTimeoutError(
                    f"Request {request_id} timed out after {duration_ms:.1f}ms",
                    context=ErrorContext(
                        operation="api_timeout",
                        component="openrouter_client",
                        additional_data={
                            "request_id": request_id,
                            "model": model,
                            "duration_ms": duration_ms,
                            "timeout": self.timeout
                        }
                    ),
                    original_exception=e
                )
            elif isinstance(e, (httpx.ConnectError, httpx.ReadError)):
                error = NetworkException(
                    f"Network error for request {request_id}: {str(e)}",
                    error_code=ErrorCode.NETWORK_CONNECTION_FAILED,
                    context=ErrorContext(
                        operation="network_error",
                        component="openrouter_client",
                        additional_data={
                            "request_id": request_id,
                            "model": model,
                            "error_type": type(e).__name__
                        }
                    ),
                    original_exception=e
                )
            else:
                error = OpenRouterException(
                    f"Unexpected error for request {request_id}: {str(e)}",
                    error_code=ErrorCode.API_UNAVAILABLE,
                    context=ErrorContext(
                        operation="unexpected_error",
                        component="openrouter_client",
                        additional_data={
                            "request_id": request_id,
                            "model": model,
                            "error_type": type(e).__name__
                        }
                    ),
                    original_exception=e
                )
            
            # Track the error
            await error_tracker.track_error(error)
            
            logger.error(f"Chat request {request_id} failed after {duration_ms:.1f}ms: {error}")
            raise error
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on model and token usage"""
        if model not in MODEL_PRICING:
            # Default pricing if model not found
            pricing = {"prompt": 1.0, "completion": 2.0}
        else:
            pricing = MODEL_PRICING[model]

        prompt_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing[
            "completion"
        ]

        return prompt_cost + completion_cost

    @retry(**RETRY_CONFIG)
    async def _make_request(
        self, model: str, messages: List[Dict[str, str]], **params
    ) -> Tuple[str, Dict[str, Any]]:
        """Make API request with retry logic and rate limit handling"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/kaggle-red-team",
            "X-Title": "Kaggle Red Team Runner",
            "Content-Type": "application/json",
        }

        payload = {"model": model, "messages": messages, **params}

        # Use shared client instance
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout calling {model}: {e}")
            raise  # Will be retried by tenacity
        
        # Handle rate limiting with custom exception
        if response.status_code == 429:
            retry_after = int(response.headers.get("retry-after", "60"))
            logger.warning(f"Rate limited by OpenRouter. Waiting {retry_after}s before retry")
            
            # Check rate limit headers for more info
            rate_limit_requests = response.headers.get("x-ratelimit-limit-requests")
            rate_limit_tokens = response.headers.get("x-ratelimit-limit-tokens") 
            
            if rate_limit_requests:
                logger.info(f"Rate limit: {rate_limit_requests} requests")
            if rate_limit_tokens:
                logger.info(f"Rate limit: {rate_limit_tokens} tokens")
                
            # Raise custom exception with retry_after value
            raise RateLimitError(
                f"Rate limited, retry after {retry_after}s",
                retry_after=retry_after
            )
        
        # Handle other HTTP errors
        if response.status_code >= 500:
            logger.warning(f"Server error {response.status_code} from OpenRouter")
            response.raise_for_status()  # Will be retried
        
        # Non-retryable client errors
        if response.status_code >= 400:
            logger.error(f"Client error {response.status_code}: {response.text}")
            response.raise_for_status()  # Will not be retried

        data = response.json()
        # Handle both sync and async json() returns for testing
        if asyncio.iscoroutine(data):
            data = await data

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Track costs
        cost = self.calculate_cost(model, usage)
        self.total_cost += cost
        self.total_tokens += usage.get("total_tokens", 0)

        # Add cost to usage data
        usage["cost_usd"] = cost

        return content, usage

    async def chat(
        self, model: str, messages: List[Dict[str, str]], **params
    ) -> Tuple[str, Dict[str, Any]]:
        """Public interface for chat completions with enhanced error handling"""
        try:
            return await self._make_request(model, messages, **params)
        except RetryError as e:
            # Extract the original exception from retry error
            last_exception = e.last_attempt.exception() if hasattr(e, 'last_attempt') else None
            
            if isinstance(last_exception, RateLimitError):
                logger.error(f"Rate limit exceeded after all retries for {model}")
                raise Exception(f"Rate limit exceeded. Please reduce request frequency or wait before retrying.")
            elif isinstance(last_exception, httpx.TimeoutException):
                logger.error(f"Request timed out after all retries for {model}")
                raise Exception(f"Request timed out. The model {model} may be overloaded.")
            else:
                logger.error(f"All retries exhausted for {model}: {last_exception}")
                raise Exception(f"Failed after {7} attempts. Last error: {last_exception}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise Exception("Invalid API key. Please check your OPENROUTER_API_KEY.")
            elif e.response.status_code == 403:
                raise Exception(f"Access forbidden. Model {model} may not be available for your account.")
            elif e.response.status_code == 404:
                raise Exception(f"Model {model} not found. Please check the model name.")
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error calling {model}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
        }


# Module-level convenience function
_client = None


async def call_or(
    model: str, messages: List[Dict[str, str]], **params
) -> Tuple[str, Dict[str, Any]]:
    """Convenience function for OpenRouter API calls"""
    global _client
    if _client is None:
        _client = OpenRouterClient()

    return await _client.chat(model, messages, **params)


def get_client() -> OpenRouterClient:
    """Get or create the singleton client instance"""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client


def reset_client():
    """Reset the client (useful for testing)"""
    global _client
    _client = None
