"""Integration tests for async/sync database interaction"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from apps.runner.app.main import app
from apps.runner.app.store.async_db import get_db_pool, ensure_schema
from apps.runner.app.util.config import load_config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "storage": {
                "sqlite_path": f"{tmpdir}/test.db",
                "transcripts_path": f"{tmpdir}/attempts.jsonl",
                "findings_path": f"{tmpdir}/findings.jsonl",
                "reports_dir": f"{tmpdir}/reports/",
            },
            "run": {
                "target_model": "test-model",
                "cost_cap_usd": 1.0,
                "max_attempts": 10,
                "categories": ["test"],
                "max_concurrency": 2,
            },
            "providers": {
                "openrouter": {
                    "base_url": "http://test",
                }
            }
        }
        yield config


@pytest.fixture
def test_client(mock_config):
    """Create test client with mocked config"""
    with patch("apps.runner.app.main.load_config", return_value=mock_config):
        with TestClient(app) as client:
            yield client


@pytest.mark.asyncio
async def test_concurrent_async_sync_access(mock_config):
    """Test that async and sync database access don't conflict"""
    db_path = mock_config["storage"]["sqlite_path"]
    
    # Ensure schema exists
    await ensure_schema(db_path)
    
    # Get async pool
    async_pool = await get_db_pool(db_path)
    
    # Import sync database module
    from apps.runner.app.store.db import open_db, init_schema
    
    # Create sync connection
    sync_conn = open_db(db_path)
    init_schema(sync_conn)
    
    # Perform concurrent operations
    async def async_operations():
        """Perform async database operations"""
        for i in range(10):
            async with async_pool.acquire() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                    (f"async_key_{i}", f"async_value_{i}")
                )
                await conn.commit()
            await asyncio.sleep(0.01)
    
    def sync_operations():
        """Perform sync database operations"""
        cursor = sync_conn.cursor()
        for i in range(10):
            cursor.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (f"sync_key_{i}", f"sync_value_{i}")
            )
            sync_conn.commit()
    
    # Run both concurrently
    import threading
    sync_thread = threading.Thread(target=sync_operations)
    sync_thread.start()
    
    await async_operations()
    
    sync_thread.join()
    
    # Verify all operations succeeded
    async with async_pool.acquire() as conn:
        # Check async writes
        for i in range(10):
            cursor = await conn.execute(
                "SELECT value FROM state WHERE key = ?",
                (f"async_key_{i}",)
            )
            row = await cursor.fetchone()
            assert row[0] == f"async_value_{i}"
        
        # Check sync writes
        for i in range(10):
            cursor = await conn.execute(
                "SELECT value FROM state WHERE key = ?",
                (f"sync_key_{i}",)
            )
            row = await cursor.fetchone()
            assert row[0] == f"sync_value_{i}"
    
    # Cleanup
    sync_conn.close()
    await async_pool.close()


@pytest.mark.asyncio
async def test_control_endpoints_with_async_db(test_client):
    """Test control endpoints use async database correctly"""
    # Test pause endpoint
    response = test_client.post("/control/pause")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["not_running", "paused", "already_paused"]
    
    # Test resume endpoint
    response = test_client.post("/control/resume")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["cannot_resume", "resumed", "already_running"]
    
    # Test stop endpoint
    response = test_client.post("/control/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["stopped", "already_stopped"]


@pytest.mark.asyncio
async def test_run_endpoint_with_async_orchestrator(test_client):
    """Test /runs endpoint with async orchestrator"""
    # Mock the orchestrator to avoid actual execution
    with patch("apps.runner.app.async_orchestrator.AsyncOrchestrator.run") as mock_run:
        mock_run.return_value = {
            "total_attempts": 10,
            "successful_attempts": 8,
            "failed_attempts": 2,
            "findings_count": 3,
            "clusters_count": 2,
            "total_cost": 0.5,
        }
        
        # Start a run
        response = test_client.post("/runs", json={
            "run_id": "test-run",
            "max_attempts": 10,
            "dry_run": True,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["run_id"] == "test-run"
        
        # Wait a bit for background task to start
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_database_pool_lifecycle(test_client):
    """Test database pool is properly managed through app lifecycle"""
    # Access the app state
    assert hasattr(app.state, "db_pool")
    assert app.state.db_pool is not None
    
    # Make some requests that use the pool
    response = test_client.get("/status")
    assert response.status_code == 200
    
    response = test_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openrouter_client_cleanup():
    """Test OpenRouter client cleanup on shutdown"""
    from apps.runner.app.providers.openrouter import OpenRouterClient
    
    client = OpenRouterClient(api_key="test-key")
    
    # Create HTTP client
    http_client = await client._get_client()
    assert http_client is not None
    
    # Close should clean up
    await client.close()
    assert client._client is None
    
    # Getting client again should create new one
    http_client2 = await client._get_client()
    assert http_client2 is not None
    assert http_client2 != http_client
    
    # Cleanup
    await client.close()


@pytest.mark.asyncio
async def test_error_handling_in_async_db(mock_config):
    """Test error handling in async database operations"""
    db_path = mock_config["storage"]["sqlite_path"]
    
    # Get pool
    pool = await get_db_pool(db_path)
    
    # Test handling of bad SQL
    async with pool.acquire() as conn:
        with pytest.raises(aiosqlite.OperationalError):
            await conn.execute("SELECT * FROM non_existent_table")
    
    # Pool should still be usable after error
    async with pool.acquire() as conn:
        result = await conn.execute("SELECT 1")
        row = await result.fetchone()
        assert row[0] == 1
    
    await pool.close()


@pytest.mark.asyncio
async def test_state_consistency_across_connections(mock_config):
    """Test state consistency across multiple connections"""
    db_path = mock_config["storage"]["sqlite_path"]
    await ensure_schema(db_path)
    
    pool = await get_db_pool(db_path)
    
    # Write with one connection
    async with pool.acquire() as conn1:
        await conn1.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            ("test_key", "test_value")
        )
        await conn1.commit()
    
    # Read with another connection
    async with pool.acquire() as conn2:
        cursor = await conn2.execute(
            "SELECT value FROM state WHERE key = ?",
            ("test_key",)
        )
        row = await cursor.fetchone()
        assert row[0] == "test_value"
    
    await pool.close()


@pytest.mark.asyncio
async def test_concurrent_pool_operations(mock_config):
    """Test concurrent operations on the pool"""
    db_path = mock_config["storage"]["sqlite_path"]
    await ensure_schema(db_path)
    
    pool = await get_db_pool(db_path)
    
    async def worker(worker_id):
        """Worker that performs database operations"""
        for i in range(5):
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                    (f"worker_{worker_id}_key_{i}", f"value_{i}")
                )
                await conn.commit()
                await asyncio.sleep(0.01)
        return worker_id
    
    # Run multiple workers concurrently
    workers = [worker(i) for i in range(5)]
    results = await asyncio.gather(*workers)
    
    assert results == list(range(5))
    
    # Verify all writes succeeded
    async with pool.acquire() as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM state")
        row = await cursor.fetchone()
        assert row[0] == 25  # 5 workers * 5 writes each
    
    await pool.close()