# ABOUTME: Async SQLite database operations using aiosqlite with connection pooling
# ABOUTME: Provides thread-safe async database access with proper connection management

import aiosqlite
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List, AsyncIterator
import json
from pathlib import Path
import logging

from apps.runner.app.util.schemas import RunStatus as RunState

logger = logging.getLogger(__name__)


class AsyncDatabase:
    """Simple async database wrapper for backward compatibility"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = AsyncDatabasePool(db_path)
    
    async def initialize(self):
        """Initialize the database"""
        await self.pool.initialize()
    
    async def close(self):
        """Close the database"""
        await self.pool.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        async with self.pool.acquire() as conn:
            yield conn


class AsyncDatabasePool:
    """Async database connection pool for SQLite with proper lifecycle management"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: List[aiosqlite.Connection] = []
        self._semaphore = asyncio.Semaphore(pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the connection pool"""
        async with self._lock:
            if self._initialized:
                return
            
            # Create initial connections
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                self._pool.append(conn)
            
            self._initialized = True
            logger.info(f"Database pool initialized with {self.pool_size} connections")
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection with optimized settings"""
        conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        
        # Enable WAL mode for concurrent access
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA journal_size_limit=67108864")  # 64MB limit
        await conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        
        # Enable row factory for dict-like access
        conn.row_factory = aiosqlite.Row
        
        return conn
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool with retry logic"""
        if not self._initialized:
            await self.initialize()
        
        await self._semaphore.acquire()
        conn = None
        max_retries = 3
        retry_count = 0
        
        try:
            while retry_count < max_retries:
                try:
                    async with self._lock:
                        if self._pool:
                            conn = self._pool.pop()
                        else:
                            # Pool exhausted, create new connection
                            logger.debug("Pool exhausted, creating new connection")
                            conn = await self._create_connection()
                    
                    # Verify connection is still alive
                    try:
                        await conn.execute("SELECT 1")
                        break  # Connection is good
                    except Exception as e:
                        logger.warning(f"Dead connection detected: {e}")
                        # Connection dead, close and retry
                        try:
                            await conn.close()
                        except Exception:
                            pass  # Ignore close errors
                        conn = None
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise ConnectionError(f"Failed to get valid connection after {max_retries} attempts")
                        await asyncio.sleep(0.1 * retry_count)  # Exponential backoff
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise ConnectionError(f"Failed to acquire connection: {e}")
                    await asyncio.sleep(0.1 * retry_count)
            
            if conn is None:
                conn = await self._create_connection()
            
            yield conn
            
        except Exception as e:
            logger.error(f"Error in connection acquisition: {e}")
            raise
        finally:
            # Always return connection to pool or close it
            if conn:
                try:
                    # Test if connection is still valid before returning to pool
                    await conn.execute("SELECT 1")
                    async with self._lock:
                        if len(self._pool) < self.pool_size:
                            self._pool.append(conn)
                        else:
                            # Pool full, close excess connection
                            await conn.close()
                except Exception:
                    # Connection is bad, just close it
                    try:
                        await conn.close()
                    except Exception:
                        pass  # Ignore close errors
            self._semaphore.release()
    
    async def close(self) -> None:
        """Close all connections in the pool"""
        async with self._lock:
            for conn in self._pool:
                try:
                    await conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self._pool.clear()
            self._initialized = False
            logger.info("Database pool closed")


# Global pool instance
_db_pool: Optional[AsyncDatabasePool] = None


async def get_db_pool(db_path: str = "data/state.db") -> AsyncDatabasePool:
    """Get or create the global database pool"""
    global _db_pool
    if _db_pool is None:
        _db_pool = AsyncDatabasePool(db_path)
        await _db_pool.initialize()
    return _db_pool


async def init_schema(conn: aiosqlite.Connection) -> None:
    """Initialize database schema. Idempotent - safe to run multiple times."""
    
    # Runs table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_model TEXT NOT NULL,
            categories TEXT NOT NULL,  -- JSON array
            max_attempts INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'PENDING',
            cost_so_far REAL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON
        )
    """)
    
    # Tasks table (individual test configurations)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            template_id TEXT NOT NULL,
            mutator_chain TEXT NOT NULL,  -- JSON array
            seed INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'PENDING',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id),
            UNIQUE(run_id, template_id, mutator_chain, seed)
        )
    """)
    
    # Attempts table (actual test executions)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT,
            model TEXT NOT NULL,
            usage TEXT,  -- JSON
            cost REAL,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        )
    """)
    
    # Evaluations table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            is_vulnerable BOOLEAN NOT NULL,
            severity TEXT,
            category TEXT,
            heuristic_score REAL,
            judge_score REAL,
            combined_score REAL NOT NULL,
            rationale TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (attempt_id) REFERENCES attempts(id),
            UNIQUE(attempt_id)
        )
    """)
    
    # Findings table (deduplicated vulnerabilities)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            cluster_id TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            representative_prompt TEXT NOT NULL,
            representative_response TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            occurrences INTEGER DEFAULT 1,
            metadata TEXT,  -- JSON
            FOREIGN KEY (run_id) REFERENCES runs(id),
            UNIQUE(run_id, cluster_id)
        )
    """)
    
    # State table (key-value store)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for performance
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_task_id ON attempts(task_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_attempt_id ON evaluations(attempt_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings(run_id)")
    
    await conn.commit()


async def get_state(conn: aiosqlite.Connection, key: str) -> Optional[str]:
    """Get a value from the state table"""
    async with conn.execute(
        "SELECT value FROM state WHERE key = ?", (key,)
    ) as cursor:
        row = await cursor.fetchone()
        return row["value"] if row else None


async def set_state(conn: aiosqlite.Connection, key: str, value: str) -> None:
    """Set a value in the state table"""
    await conn.execute(
        """INSERT OR REPLACE INTO state (key, value, updated_at) 
           VALUES (?, ?, CURRENT_TIMESTAMP)""",
        (key, value)
    )
    await conn.commit()


async def ensure_schema(db_path: str = "data/state.db") -> None:
    """Ensure database schema is initialized"""
    pool = await get_db_pool(db_path)
    async with pool.acquire() as conn:
        await init_schema(conn)