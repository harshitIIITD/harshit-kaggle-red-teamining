# ABOUTME: Tests for OpenRouter async client with retries and cost tracking
# ABOUTME: Validates API interactions, retry logic, headers, and usage capture

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from apps.runner.app.providers.openrouter import call_or, OpenRouterClient, reset_client


@pytest.fixture(autouse=True)
def reset_client_fixture():
    """Reset client before each test"""
    reset_client()
    yield
    reset_client()


@pytest.mark.asyncio
async def test_success_call_returns_content_and_usage():
    """Test successful API call returns content and usage stats"""
    mock_response = {
        "choices": [{"message": {"content": "Test response content"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response_obj = AsyncMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.headers = {}
            mock_response_obj.raise_for_status = AsyncMock()

            mock_post.return_value = mock_response_obj

            content, usage = await call_or(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert content == "Test response content"
            assert usage["prompt_tokens"] == 100
            assert usage["completion_tokens"] == 50
            assert usage["total_tokens"] == 150


@pytest.mark.asyncio
async def test_retry_on_429_then_success():
    """Test retry logic on rate limit (429) then succeeds"""
    mock_429_response = AsyncMock()
    mock_429_response.status_code = 429
    mock_429_response.json.return_value = {"error": "Rate limit exceeded"}
    mock_429_response.headers = {"retry-after": "1"}
    mock_429_response.request = MagicMock()
    mock_429_response.raise_for_status = AsyncMock()

    mock_success_response = AsyncMock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {
        "choices": [{"message": {"content": "Success after retry"}}],
        "usage": {"total_tokens": 100},
    }
    mock_success_response.headers = {}
    mock_success_response.raise_for_status = AsyncMock()

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        with patch("httpx.AsyncClient.post") as mock_post:
            # First call returns 429, second succeeds
            mock_post.side_effect = [mock_429_response, mock_success_response]

            # Patch sleep to avoid waiting in tests
            with patch("asyncio.sleep", new_callable=AsyncMock):
                content, usage = await call_or(
                    model="openai/gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                )

                assert content == "Success after retry"
                assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_headers_include_title_referer():
    """Test that API requests include proper headers for OpenRouter"""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response_obj = AsyncMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = {
                "choices": [{"message": {"content": "Test"}}],
                "usage": {"total_tokens": 10},
            }
            mock_response_obj.headers = {}
            mock_response_obj.raise_for_status = AsyncMock()

            mock_post.return_value = mock_response_obj

            await call_or(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
            )

            # Check that headers were set correctly
            call_args = mock_post.call_args
            headers = call_args.kwargs.get("headers", {})

            assert "HTTP-Referer" in headers
            assert "X-Title" in headers
            assert headers["X-Title"] == "Kaggle Red Team Runner"


@pytest.mark.asyncio
async def test_cost_calculation():
    """Test cost calculation based on usage"""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        client = OpenRouterClient()

        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}

        # Test with a model that has known pricing
        cost = client.calculate_cost("openai/gpt-3.5-turbo", usage)

        # Cost should be non-negative
        assert cost >= 0

        # Cost should scale with tokens
        usage_double = {
            "prompt_tokens": 2000,
            "completion_tokens": 1000,
            "total_tokens": 3000,
        }
        cost_double = client.calculate_cost("openai/gpt-3.5-turbo", usage_double)
        assert cost_double > cost


@pytest.mark.asyncio
async def test_timeout_enforced():
    """Test that requests timeout after configured duration"""
    with patch("httpx.AsyncClient.post") as mock_post:
        # Simulate a timeout
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(Exception):  # Will be wrapped by tenacity
            await call_or(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                timeout=1,  # 1 second timeout
            )


@pytest.mark.asyncio
async def test_api_key_from_env():
    """Test that API key is loaded from environment"""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"}):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response_obj = AsyncMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = {
                "choices": [{"message": {"content": "Test"}}],
                "usage": {"total_tokens": 10},
            }
            mock_response_obj.headers = {}
            mock_response_obj.raise_for_status = AsyncMock()

            mock_post.return_value = mock_response_obj

            await call_or(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
            )

            # Check Authorization header
            call_args = mock_post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-key-123"
