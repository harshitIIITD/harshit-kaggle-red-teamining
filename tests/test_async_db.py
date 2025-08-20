"""Comprehensive tests for async database module"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import aiosqlite

from apps.runner.app.store.async_db import (
    AsyncDatabasePool,
    get_db_pool,
    init_schema,
    get_state,
    set_state,
    ensure_schema,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture
async def db_pool(temp_db_path):
    """Create a test database pool"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=3)
    await pool.initialize()
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_pool_initialization(temp_db_path):
    """Test database pool initialization"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=5)
    
    # Pool should not be initialized yet
    assert not pool._initialized
    assert len(pool._pool) == 0
    
    # Initialize pool
    await pool.initialize()
    
    # Pool should be initialized with correct size
    assert pool._initialized
    assert len(pool._pool) == 5
    
    # Cleanup
    await pool.close()


@pytest.mark.asyncio
async def test_pool_acquire_and_release(db_pool):
    """Test acquiring and releasing connections from pool"""
    initial_size = len(db_pool._pool)
    
    # Acquire a connection
    async with db_pool.acquire() as conn:
        assert conn is not None
        # Pool should have one less connection
        assert len(db_pool._pool) == initial_size - 1
        
        # Connection should work
        result = await conn.execute("SELECT 1")
        row = await result.fetchone()
        assert row[0] == 1
    
    # Connection should be returned to pool
    assert len(db_pool._pool) == initial_size


@pytest.mark.asyncio
async def test_pool_concurrent_access(db_pool):
    """Test concurrent access to the pool"""
    async def use_connection(pool, id):
        async with pool.acquire() as conn:
            # Simulate some work
            await conn.execute("SELECT ?", (id,))
            await asyncio.sleep(0.01)
            return id
    
    # Run multiple concurrent tasks
    tasks = [use_connection(db_pool, i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All tasks should complete successfully
    assert results == list(range(10))


@pytest.mark.asyncio
async def test_pool_exhaustion_handling(temp_db_path):
    """Test behavior when pool is exhausted"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=2)
    await pool.initialize()
    
    try:
        # Test that we can acquire up to pool_size connections concurrently
        async def use_connection(pool, delay=0.1):
            async with pool.acquire() as conn:
                # Hold connection briefly
                await asyncio.sleep(delay)
                result = await conn.execute("SELECT 1")
                row = await result.fetchone()
                return row[0]
        
        # Run pool_size + 1 concurrent tasks
        # The extra task should wait for a connection to be released
        tasks = [use_connection(pool, 0.1) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert all(r == 1 for r in results)
        assert len(results) == 3
            
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_dead_connection_recovery(db_pool):
    """Test recovery from dead connections"""
    # Get a connection and make it "dead"
    async with db_pool.acquire() as conn:
        # Mock the execute method to raise an exception
        original_execute = conn.execute
        conn.execute = AsyncMock(side_effect=aiosqlite.OperationalError("Connection lost"))
        
        # Pool should detect dead connection and create new one
        # This happens internally in acquire()
    
    # Next acquire should work with new connection
    async with db_pool.acquire() as conn:
        result = await conn.execute("SELECT 1")
        row = await result.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_retry_logic(temp_db_path):
    """Test retry logic when acquiring connections"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=1)
    await pool.initialize()
    
    # Mock _create_connection to fail first 2 times
    call_count = 0
    original_create = pool._create_connection
    
    async def mock_create():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Mock connection error")
        return await original_create()
    
    with patch.object(pool, '_create_connection', mock_create):
        # Clear the pool to force new connection creation
        pool._pool.clear()
        
        # Should retry and eventually succeed
        async with pool.acquire() as conn:
            assert conn is not None
            result = await conn.execute("SELECT 1")
            row = await result.fetchone()
            assert row[0] == 1
    
    await pool.close()


@pytest.mark.asyncio
async def test_init_schema(db_pool):
    """Test schema initialization"""
    async with db_pool.acquire() as conn:
        await init_schema(conn)
        
        # Check that tables exist
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        
        expected_tables = {'runs', 'tasks', 'attempts', 'evaluations', 'findings', 'state'}
        assert expected_tables.issubset(tables)


@pytest.mark.asyncio
async def test_state_operations(db_pool):
    """Test state get/set operations"""
    async with db_pool.acquire() as conn:
        await init_schema(conn)
        
        # Test setting state
        await set_state(conn, "TEST_KEY", "test_value")
        
        # Test getting state
        value = await get_state(conn, "TEST_KEY")
        assert value == "test_value"
        
        # Test getting non-existent key
        value = await get_state(conn, "NON_EXISTENT")
        assert value is None
        
        # Test updating state
        await set_state(conn, "TEST_KEY", "updated_value")
        value = await get_state(conn, "TEST_KEY")
        assert value == "updated_value"


@pytest.mark.asyncio
async def test_pool_cleanup_on_error(temp_db_path):
    """Test that pool properly cleans up on errors"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=2)
    await pool.initialize()
    
    # Force an error during acquire
    with patch.object(pool, '_create_connection', side_effect=Exception("Test error")):
        pool._pool.clear()  # Clear pool to force creation
        
        with pytest.raises(ConnectionError):
            async with pool.acquire() as conn:
                pass  # Should not reach here
    
    # Pool should still be usable after error
    async with pool.acquire() as conn:
        result = await conn.execute("SELECT 1")
        row = await result.fetchone()
        assert row[0] == 1
    
    await pool.close()


@pytest.mark.asyncio
async def test_global_pool_singleton(temp_db_path):
    """Test that get_db_pool returns singleton"""
    pool1 = await get_db_pool(temp_db_path)
    pool2 = await get_db_pool(temp_db_path)
    
    assert pool1 is pool2
    
    await pool1.close()


@pytest.mark.asyncio
async def test_ensure_schema(temp_db_path):
    """Test ensure_schema function"""
    await ensure_schema(temp_db_path)
    
    # Get pool and check schema
    pool = await get_db_pool(temp_db_path)
    async with pool.acquire() as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        
        expected_tables = {'runs', 'tasks', 'attempts', 'evaluations', 'findings', 'state'}
        assert expected_tables.issubset(tables)
    
    await pool.close()


@pytest.mark.asyncio
async def test_wal_mode_enabled(db_pool):
    """Test that WAL mode is enabled"""
    async with db_pool.acquire() as conn:
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_connection_timeout_handling(temp_db_path):
    """Test handling of connection timeouts"""
    pool = AsyncDatabasePool(temp_db_path, pool_size=1)
    await pool.initialize()
    
    # Test that tasks wait for connections when pool is exhausted
    results = []
    
    async def hold_connection(duration):
        async with pool.acquire() as conn:
            await asyncio.sleep(duration)
            result = await conn.execute("SELECT 1")
            row = await result.fetchone()
            results.append(row[0])
    
    # Start two tasks - second should wait for first to release
    task1 = asyncio.create_task(hold_connection(0.1))
    await asyncio.sleep(0.01)  # Ensure task1 gets connection first
    task2 = asyncio.create_task(hold_connection(0.1))
    
    # Both should complete
    await asyncio.gather(task1, task2)
    assert results == [1, 1]
    
    await pool.close()


@pytest.mark.asyncio 
async def test_pool_close_idempotent(db_pool):
    """Test that closing pool multiple times is safe"""
    await db_pool.close()
    await db_pool.close()  # Should not raise
    
    assert not db_pool._initialized
    assert len(db_pool._pool) == 0