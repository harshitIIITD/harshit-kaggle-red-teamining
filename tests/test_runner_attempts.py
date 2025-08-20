# ABOUTME: Test suite for the attempt runner functionality that executes prompts against target models
# ABOUTME: Validates JSONL structure, error handling, and PII redaction capabilities

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from datetime import datetime, timezone

from apps.runner.app.agents.tester import run_attempt, classify_error
from apps.runner.app.store.files import append_jsonl, read_jsonl_lines


class TestAttemptRunner:
    
    @pytest.mark.asyncio
    async def test_append_jsonl_shape_and_fields(self):
        """Test that JSONL records have correct structure and required fields"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            # Create test attempt record
            record = {
                "id": "test-001",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": "Test prompt",
                "model": "test-model",
                "response": "Test response",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "cost_usd": 0.001,
                "error": None,
                "metadata": {
                    "template_id": "test-template",
                    "mutator_chain": ["uppercase"],
                    "seed": 42
                }
            }
            
            # Append to JSONL
            append_jsonl(jsonl_path, record)
            
            # Read back and verify
            lines = read_jsonl_lines(jsonl_path)
            assert len(lines) == 1
            
            parsed = json.loads(lines[0])
            assert parsed["id"] == "test-001"
            assert "timestamp" in parsed
            assert parsed["prompt"] == "Test prompt"
            assert parsed["response"] == "Test response"
            assert parsed["usage"]["total_tokens"] == 30
            assert parsed["cost_usd"] == 0.001
            assert parsed["metadata"]["seed"] == 42
    
    @pytest.mark.asyncio
    async def test_error_marked_failed_temp_vs_perm(self):
        """Test that errors are classified as temporary vs permanent correctly"""
        # Test temporary errors (retryable)
        assert classify_error({"status": 429, "message": "Rate limited"}) == "temporary"
        assert classify_error({"status": 503, "message": "Service unavailable"}) == "temporary"
        assert classify_error({"status": 500, "message": "Internal error"}) == "temporary"
        
        # Test permanent errors (non-retryable)
        assert classify_error({"status": 400, "message": "Invalid request"}) == "permanent"
        assert classify_error({"status": 401, "message": "Unauthorized"}) == "permanent"
        assert classify_error({"status": 404, "message": "Not found"}) == "permanent"
        
        # Test unknown errors default to temporary
        assert classify_error({"status": 999, "message": "Unknown"}) == "temporary"
        assert classify_error(None) == "temporary"
    
    @pytest.mark.asyncio
    async def test_run_attempt_success(self):
        """Test successful attempt execution"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            # Mock the OpenRouter call (returns content, usage)
            mock_response = ("Test output", {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            })
            
            with patch('apps.runner.app.agents.tester.call_or', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = mock_response
                
                result = await run_attempt(
                    prompt="What is 2+2?",
                    model="meta-llama/llama-3.1-8b-instruct",
                    jsonl_path=jsonl_path,
                    template_id="math-test",
                    mutator_chain=["none"],
                    seed=12345
                )
                
                assert result["success"] is True
                assert result["response"] == "Test output"
                assert result["usage"]["total_tokens"] == 150
                assert result["error"] is None
                
                # Verify JSONL was written
                lines = read_jsonl_lines(jsonl_path)
                assert len(lines) == 1
                parsed = json.loads(lines[0])
                assert parsed["prompt"] == "What is 2+2?"
                assert parsed["response"] == "Test output"
    
    @pytest.mark.asyncio
    async def test_run_attempt_with_error(self):
        """Test attempt execution with error handling"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            # Mock the OpenRouter call to raise an error
            with patch('apps.runner.app.agents.tester.call_or', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = Exception("API Error: Rate limited")
                
                result = await run_attempt(
                    prompt="Test prompt",
                    model="test-model",
                    jsonl_path=jsonl_path,
                    template_id="test",
                    mutator_chain=[],
                    seed=1
                )
                
                assert result["success"] is False
                assert result["response"] is None
                assert "API Error" in result["error"]["message"]
                
                # Verify error was logged to JSONL
                lines = read_jsonl_lines(jsonl_path)
                assert len(lines) == 1
                parsed = json.loads(lines[0])
                assert parsed["error"] is not None
                assert "API Error" in parsed["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_pii_redaction(self):
        """Test that PII is redacted before writing to JSONL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            # Response containing PII-like patterns
            mock_response = (
                "My email is john.doe@example.com and phone is 555-1234-5678. "
                "SSN: 123-45-6789 and credit card 4111-1111-1111-1111.",
                {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
            )
            
            with patch('apps.runner.app.agents.tester.call_or', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = mock_response
                
                await run_attempt(
                    prompt="Extract personal info",
                    model="test-model",
                    jsonl_path=jsonl_path,
                    template_id="test",
                    mutator_chain=[],
                    seed=1
                )
                
                # Read JSONL and verify PII was redacted
                lines = read_jsonl_lines(jsonl_path)
                parsed = json.loads(lines[0])
                
                # Check that PII patterns are redacted
                assert "john.doe@example.com" not in parsed["response"]
                assert "[EMAIL]" in parsed["response"] or "***" in parsed["response"]
                assert "555-1234-5678" not in parsed["response"]
                assert "123-45-6789" not in parsed["response"]
                assert "4111-1111-1111-1111" not in parsed["response"]
    
    def test_jsonl_rotation_safety(self):
        """Test that JSONL file handles rotation safely"""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            # Write multiple records
            for i in range(5):
                record = {
                    "id": f"test-{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": f"Prompt {i}",
                    "response": f"Response {i}"
                }
                append_jsonl(jsonl_path, record)
            
            # Verify all records are present and valid
            lines = read_jsonl_lines(jsonl_path)
            assert len(lines) == 5
            
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["id"] == f"test-{i}"
                assert parsed["prompt"] == f"Prompt {i}"
    
    def test_concurrent_append_safety(self):
        """Test that concurrent appends to JSONL are safe"""
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "attempts.jsonl"
            
            def write_records(thread_id):
                for i in range(10):
                    record = {
                        "id": f"thread-{thread_id}-record-{i}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": f"Data from thread {thread_id}"
                    }
                    append_jsonl(jsonl_path, record)
            
            # Create multiple threads writing concurrently
            threads = []
            for tid in range(5):
                t = threading.Thread(target=write_records, args=(tid,))
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
            
            # Verify all records were written
            lines = read_jsonl_lines(jsonl_path)
            assert len(lines) == 50  # 5 threads * 10 records each
            
            # Verify each line is valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert "id" in parsed
                assert "timestamp" in parsed