# ABOUTME: Ollama async client for local LLM inference
# ABOUTME: Handles local API calls without authentication or cost tracking

import httpx
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Custom exception for connection errors
class OllamaConnectionError(Exception):
    """Raised when Ollama service is not available"""
    pass

# Retry configuration for local connection issues
RETRY_CONFIG = {
    "stop": stop_after_attempt(3),  # Fewer retries for local service
    "wait": wait_exponential(multiplier=1, min=1, max=10),
    "retry": retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, OllamaConnectionError)),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
}

# Model mapping from OpenRouter names to Ollama local names
MODEL_MAPPING = {
    "openai/gpt-3.5-turbo": "llama3",
    "openai/gpt-4": "llama3",
    "openai/gpt-4-turbo": "llama3",
    "openai/gpt-oss-20b": "llama3",  # Main target model
    "meta-llama/llama-3.1-8b-instruct": "llama3",
    "meta-llama/llama-3.1-70b-instruct": "llama3:70b",
    "anthropic/claude-3-haiku": "llama3",
    "anthropic/claude-3-sonnet": "llama3",
}


@dataclass
class OllamaClient:
    """Async client for Ollama local API without authentication or cost tracking"""

    base_url: str = "http://localhost:11434"
    timeout: int = 60
    total_tokens: int = 0  # Keep for compatibility, no cost tracking
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False)
    _client_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        # No API key required for local Ollama
        pass
    
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

    def _map_model_name(self, model: str) -> str:
        """Map OpenRouter model names to Ollama local model names"""
        return MODEL_MAPPING.get(model, "llama3")  # Default to llama3

    @retry(**RETRY_CONFIG)
    async def _make_request(
        self, model: str, messages: List[Dict[str, str]], **params
    ) -> Tuple[str, Dict[str, Any]]:
        """Make API request to Ollama with retry logic"""

        # Map to local model name
        local_model = self._map_model_name(model)
        
        # Ollama API format
        payload = {
            "model": local_model,
            "messages": messages,
            "stream": False  # Get complete response
        }
        
        # Add any additional parameters
        if params:
            payload.update(params)

        # Use shared client instance
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/api/chat", 
                json=payload
            )
        except httpx.ConnectError as e:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise OllamaConnectionError(f"Ollama service not available at {self.base_url}")
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout calling Ollama model {local_model}: {e}")
            raise  # Will be retried by tenacity
        
        # Handle model not found errors (don't retry these)
        if response.status_code == 404:
            logger.error(f"Ollama model {local_model} not found: {response.text}")
            response.raise_for_status()  # This will raise HTTPStatusError, not retried
        
        # Handle other HTTP errors
        if response.status_code >= 400:
            logger.error(f"Ollama error {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()
        # Handle both sync and async json() returns for testing
        if asyncio.iscoroutine(data):
            data = await data

        # Extract content from Ollama response format
        content = data["message"]["content"]
        
        # Create usage data for compatibility (Ollama doesn't provide exact token counts)
        # Estimate tokens for compatibility with existing code
        estimated_prompt_tokens = sum(len(msg.get("content", "").split()) for msg in messages)
        estimated_completion_tokens = len(content.split())
        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
        
        usage = {
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_completion_tokens,
            "total_tokens": estimated_total_tokens,
            "cost_usd": 0.0  # No cost for local inference
        }

        # Track tokens for compatibility
        self.total_tokens += estimated_total_tokens

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
            
            if isinstance(last_exception, OllamaConnectionError):
                logger.error(f"Ollama connection failed after all retries for {model}")
                raise Exception(f"Cannot connect to Ollama service. Please ensure Ollama is running at {self.base_url}")
            elif isinstance(last_exception, httpx.TimeoutException):
                logger.error(f"Request timed out after all retries for {model}")
                raise Exception(f"Request timed out. Ollama may be overloaded or the model may be loading.")
            else:
                logger.error(f"All retries exhausted for {model}: {last_exception}")
                raise Exception(f"Failed after 3 attempts. Last error: {last_exception}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                local_model = self._map_model_name(model)
                raise Exception(f"Model {local_model} not found in Ollama. Please install it with: ollama pull {local_model}")
            else:
                raise Exception(f"Ollama service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama with model {model}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics (no cost for local inference)"""
        return {
            "total_cost_usd": 0.0,  # Always 0 for local inference
            "total_tokens": self.total_tokens,
        }


# Module-level convenience function
_client = None


async def call_ollama(
    model: str, messages: List[Dict[str, str]], **params
) -> Tuple[str, Dict[str, Any]]:
    """Convenience function for Ollama API calls"""
    global _client
    if _client is None:
        _client = OllamaClient()

    return await _client.chat(model, messages, **params)


def get_client() -> OllamaClient:
    """Get or create the singleton client instance"""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def reset_client():
    """Reset the client (useful for testing)"""
    global _client
    _client = None