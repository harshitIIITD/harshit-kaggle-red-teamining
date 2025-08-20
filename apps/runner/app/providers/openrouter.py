# ABOUTME: OpenRouter async client with retry logic and cost tracking
# ABOUTME: Handles API calls, retries on failures, and tracks token usage/costs

import os
import httpx
import asyncio
from typing import Dict, List, Tuple, Any, Optional
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

logger = logging.getLogger(__name__)

# Custom exception for rate limiting
class RateLimitError(Exception):
    """Raised when rate limited"""
    def __init__(self, message: str, retry_after: int = 5):
        super().__init__(message)
        self.retry_after = retry_after

# Custom wait strategy that respects retry_after for rate limits
class wait_rate_limit:
    """Wait strategy that uses retry_after from RateLimitError or exponential backoff"""
    def __init__(self, exponential_base=2, exponential_max=120):
        self.exponential = wait_exponential(multiplier=exponential_base, min=1, max=exponential_max)
    
    def __call__(self, retry_state):
        # Check if the last exception was a RateLimitError with retry_after
        if retry_state.outcome and retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            if isinstance(exception, RateLimitError) and hasattr(exception, 'retry_after'):
                # Use the retry_after value from the API
                wait_time = exception.retry_after
                logger.info(f"Using retry_after value: {wait_time}s")
                return wait_time
        
        # Fall back to exponential backoff for other errors
        return self.exponential(retry_state)

# Retry configuration with custom logic for rate limits
def should_retry(exception):
    """Custom retry logic for different exception types"""
    if isinstance(exception, RateLimitError):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 429 (rate limit) and 5xx errors
        return exception.response.status_code in [429, 500, 502, 503, 504]
    if isinstance(exception, httpx.TimeoutException):
        return True
    return False

RETRY_CONFIG = {
    "stop": stop_after_attempt(7),  # More attempts for rate limiting
    "wait": wait_rate_limit(exponential_base=2, exponential_max=120),  # Custom wait strategy
    "retry": retry_if_exception_type((RateLimitError, httpx.HTTPStatusError, httpx.TimeoutException)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
}

# Model pricing per 1M tokens (approximate, update as needed)
MODEL_PRICING = {
    "openai/gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
    "openai/gpt-4": {"prompt": 30.0, "completion": 60.0},
    "openai/gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "openai/gpt-oss-20b": {"prompt": 2.0, "completion": 6.0},  # Placeholder
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.07, "completion": 0.07},
    "meta-llama/llama-3.1-70b-instruct": {"prompt": 0.59, "completion": 0.79},
    "anthropic/claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    "anthropic/claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
}


@dataclass
class OpenRouterClient:
    """Async client for OpenRouter API with retry and cost tracking"""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    timeout: int = 60
    total_cost: float = 0.0
    total_tokens: int = 0
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (singleton pattern)"""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
                    )
        return self._client
    
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
