# ABOUTME: Tests for Ollama async client for local inference
# ABOUTME: Validates API interactions, retry logic, and local model handling

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from apps.runner.app.providers.ollama import call_ollama, OllamaClient, reset_client, OllamaConnectionError


@pytest.fixture(autouse=True)
def reset_client_fixture():
    """Reset client before each test"""
    reset_client()
    yield
    reset_client()


@pytest.mark.asyncio
async def test_success_call_returns_content_and_usage():
    """Test successful API call returns content and usage stats"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Test response"},
        }
        mock_response_obj.raise_for_status = AsyncMock()

        mock_post.return_value = mock_response_obj

        content, usage = await call_ollama(
            model="llama3",
            messages=[{"role": "user", "content": "Test message"}],
        )

        assert content == "Test response"
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["cost_usd"] == 0.0  # No cost for local inference
        assert usage["total_tokens"] > 0


@pytest.mark.asyncio
async def test_retry_on_connection_error_then_success():
    """Test retry logic on connection error then succeeds"""
    with patch("httpx.AsyncClient.post") as mock_post:
        # First call fails with connection error, second succeeds
        connection_error = httpx.ConnectError("Connection failed")
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Success after retry"}
        }
        mock_response_obj.raise_for_status = AsyncMock()

        mock_post.side_effect = [connection_error, mock_response_obj]

        content, usage = await call_ollama(
            model="llama3",
            messages=[{"role": "user", "content": "Test"}],
        )

        assert content == "Success after retry"
        assert mock_post.call_count == 2  # One retry


@pytest.mark.asyncio
async def test_model_mapping():
    """Test that OpenRouter model names are mapped to Ollama models"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Test"}
        }
        mock_response_obj.raise_for_status = AsyncMock()
        mock_post.return_value = mock_response_obj

        # Test mapping from OpenRouter model to Ollama model
        await call_ollama(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "Test"}],
        )

        # Check that the request was made with the mapped model name
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["model"] == "llama3"  # Should be mapped from OpenRouter name


@pytest.mark.asyncio
async def test_no_authentication_required():
    """Test that no authentication headers are required"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Test"}
        }
        mock_response_obj.raise_for_status = AsyncMock()
        mock_post.return_value = mock_response_obj

        await call_ollama(
            model="llama3",
            messages=[{"role": "user", "content": "Test"}],
        )

        # Check that no Authorization header is present
        call_args = mock_post.call_args
        assert "headers" not in call_args.kwargs or \
               "Authorization" not in call_args.kwargs.get("headers", {})


@pytest.mark.asyncio
async def test_connection_error_handling():
    """Test handling of Ollama connection errors"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(Exception) as excinfo:
            await call_ollama(
                model="llama3",
                messages=[{"role": "user", "content": "Test"}],
            )

        assert "Cannot connect to Ollama service" in str(excinfo.value)


@pytest.mark.asyncio
async def test_model_not_found_error():
    """Test handling when requested model is not installed in Ollama"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = MagicMock()  # Use MagicMock instead of AsyncMock
        mock_response_obj.status_code = 404
        mock_response_obj.text = "Model not found"
        mock_response_obj.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Model not found", request=MagicMock(), response=mock_response_obj
        )

        mock_post.return_value = mock_response_obj

        with pytest.raises(Exception) as excinfo:
            await call_ollama(
                model="nonexistent-model",
                messages=[{"role": "user", "content": "Test"}],
            )

        assert "not found in Ollama" in str(excinfo.value)
        assert "ollama pull" in str(excinfo.value)


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test that requests timeout after configured duration"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(Exception) as excinfo:
            await call_ollama(
                model="llama3",
                messages=[{"role": "user", "content": "Test"}],
            )

        assert "timed out" in str(excinfo.value)


@pytest.mark.asyncio
async def test_usage_stats_no_cost():
    """Test that usage statistics show zero cost"""
    client = OllamaClient()
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Test response"}
        }
        mock_response_obj.raise_for_status = AsyncMock()
        mock_post.return_value = mock_response_obj

        content, usage = await client.chat(
            "llama3",
            [{"role": "user", "content": "Test"}]
        )

        stats = client.get_stats()
        assert stats["total_cost_usd"] == 0.0
        assert stats["total_tokens"] > 0


@pytest.mark.asyncio
async def test_ollama_api_format():
    """Test that requests are formatted correctly for Ollama API"""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response_obj = AsyncMock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = {
            "message": {"content": "Test"}
        }
        mock_response_obj.raise_for_status = AsyncMock()
        mock_post.return_value = mock_response_obj

        await call_ollama(
            model="llama3",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7  # Test additional parameters
        )

        # Verify API call format
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        
        payload = call_args.kwargs.get("json", {})
        assert payload["model"] == "llama3"
        assert payload["messages"] == [{"role": "user", "content": "Test message"}]
        assert payload["stream"] is False
        assert payload["temperature"] == 0.7