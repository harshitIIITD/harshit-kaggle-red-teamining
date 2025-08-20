# ABOUTME: Enhanced async SQLite database operations with comprehensive error handling
# ABOUTME: Provides robust database access with connection pooling, error recovery, and monitoring

import aiosqlite
import asyncio
import time
import os
from contextlib import asynccontextmanager
from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any, List, AsyncIterator, Union, Tuple
import json
from pathlib import Path
import logging
import sqlite3
import shutil

from ..util.exceptions import (
    BaseRedTeamException, DatabaseException, ResourceException, 
    ErrorCode, ErrorSeverity, ErrorContext, error_tracker, handle_exceptions
)
from ..util.retry import database_retry_handler, RetryConfig, BackoffStrategy
from ..monitoring.health import add_custom_metric, MetricType
from ..util.schemas import RunStatus as RunState

logger = logging.getLogger(__name__)


class DatabaseHealthChecker:
    """Health checker for database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.last_check: Optional[datetime] = None
        self.consecutive_failures = 0
        
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        start_time = time.time()
        
        try:
            # Check file existence and permissions
            if not os.path.exists(self.db_path):
                return {
                    "healthy": False,
                    "error": "Database file does not exist",
                    "file_exists": False
                }
            
            if not os.access(self.db_path, os.R_OK | os.W_OK):
                return {
                    "healthy": False,
                    "error": "Insufficient file permissions",
                    "file_exists": True,
                    "readable": os.access(self.db_path, os.R_OK),
                    "writable": os.access(self.db_path, os.W_OK)
                }
            
            # Check database connectivity and integrity
            async with aiosqlite.connect(self.db_path) as conn:
                # Basic connectivity test
                await conn.execute("SELECT 1")
                
                # Check database integrity
                integrity_result = await conn.execute("PRAGMA integrity_check")
                integrity_row = await integrity_result.fetchone()
                integrity_ok = integrity_row and integrity_row[0] == "ok"
                
                # Get database info
                size_bytes = os.path.getsize(self.db_path)
                
                # Check WAL mode
                wal_result = await conn.execute("PRAGMA journal_mode")
                wal_row = await wal_result.fetchone()
                journal_mode = wal_row[0] if wal_row else "unknown"
                
                # Performance test
                perf_start = time.time()
                await conn.execute("CREATE TEMP TABLE IF NOT EXISTS health_test (id INTEGER, ts REAL)")
                await conn.execute("INSERT INTO health_test VALUES (1, ?)", (time.time(),))
                result = await conn.execute("SELECT * FROM health_test WHERE id = 1")
                await result.fetchone()
                await conn.execute("DROP TABLE health_test")
                perf_duration = (time.time() - perf_start) * 1000
                
                duration_ms = (time.time() - start_time) * 1000
                
                self.consecutive_failures = 0
                self.last_check = datetime.now(UTC)
                
                return {
                    "healthy": True,
                    "file_exists": True,
                    "readable": True,
                    "writable": True,
                    "integrity_ok": integrity_ok,
                    "size_bytes": size_bytes,
                    "journal_mode": journal_mode,
                    "response_time_ms": duration_ms,
                    "performance_test_ms": perf_duration,
                    "last_check": self.last_check.isoformat(),
                    "consecutive_failures": self.consecutive_failures
                }
                
        except Exception as e:
            self.consecutive_failures += 1
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "healthy": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "response_time_ms": duration_ms,
                "consecutive_failures": self.consecutive_failures,
                "last_check": datetime.now(UTC).isoformat()
            }


class EnhancedAsyncDatabase:
    """Enhanced async database wrapper with comprehensive error handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = EnhancedAsyncDatabasePool(db_path)
        self.health_checker = DatabaseHealthChecker(db_path)
        self._backup_manager: Optional['DatabaseBackupManager'] = None
        
    async def initialize(self):
        """Initialize the database with error handling"""
        try:
            await self.pool.initialize()
            # Initialize backup manager
            self._backup_manager = DatabaseBackupManager(self.db_path)
            logger.info(f"Initialized enhanced database: {self.db_path}")
        except Exception as e:
            error = DatabaseException(
                f"Failed to initialize database: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_FAILED,
                context=ErrorContext(
                    operation="database_initialization",
                    component="enhanced_async_database",
                    additional_data={"db_path": self.db_path}
                ),
                original_exception=e
            )
            await error_tracker.track_error(error)
            raise error
    
    async def close(self):
        """Close the database with cleanup"""
        await self.pool.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection with enhanced error handling"""
        async with self.pool.acquire() as conn:
            yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        return await self.health_checker.check_health()
    
    async def backup(self, backup_path: Optional[str] = None) -> str:
        """Create database backup"""
        if not self._backup_manager:
            raise DatabaseException(
                "Backup manager not initialized",
                error_code=ErrorCode.DB_CONNECTION_FAILED
            )
        return await self._backup_manager.create_backup(backup_path)


class DatabaseBackupManager:
    """Manages database backups and recovery"""
    
    def __init__(self, db_path: str, backup_dir: Optional[str] = None):
        self.db_path = db_path
        self.backup_dir = backup_dir or str(Path(db_path).parent / "backups")
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database"""
        try:
            if not backup_path:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                backup_filename = f"db_backup_{timestamp}.db"
                backup_path = str(Path(self.backup_dir) / backup_filename)
            
            # Create backup using SQLite's backup API
            async with aiosqlite.connect(self.db_path) as source:
                async with aiosqlite.connect(backup_path) as backup:
                    await source.backup(backup)
            
            logger.info(f"Created database backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            error = DatabaseException(
                f"Failed to create backup: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_FAILED,
                context=ErrorContext(
                    operation="database_backup",
                    component="backup_manager",
                    additional_data={
                        "source_db": self.db_path,
                        "backup_path": backup_path
                    }
                ),
                original_exception=e
            )
            await error_tracker.track_error(error)
            raise error
    
    async def restore_backup(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_path):
                raise DatabaseException(
                    f"Backup file not found: {backup_path}",
                    error_code=ErrorCode.FS_FILE_NOT_FOUND
                )
            
            # Create a backup of current database before restore
            current_backup = await self.create_backup()
            logger.info(f"Created safety backup: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Restored database from backup: {backup_path}")
            return True
            
        except Exception as e:
            error = DatabaseException(
                f"Failed to restore backup: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_FAILED,
                context=ErrorContext(
                    operation="database_restore",
                    component="backup_manager",
                    additional_data={
                        "backup_path": backup_path,
                        "target_db": self.db_path
                    }
                ),
                original_exception=e
            )
            await error_tracker.track_error(error)
            raise error
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        backup_dir = Path(self.backup_dir)
        
        if backup_dir.exists():
            for backup_file in backup_dir.glob("*.db"):
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime, UTC).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
                })
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)


# Backwards compatibility
AsyncDatabase = EnhancedAsyncDatabase


class EnhancedAsyncDatabasePool:
    """Enhanced async database connection pool with comprehensive error handling and monitoring"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self._pool: List[aiosqlite.Connection] = []
        self._semaphore = asyncio.Semaphore(pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        
        # Enhanced features
        self._connection_count = 0
        self._active_connections = 0
        self._total_acquisitions = 0
        self._failed_acquisitions = 0
        self._connection_timeouts = 0
        self._last_maintenance: Optional[datetime] = None
        self._unhealthy_connections: set = set()
        
        # Configuration
        self.connection_timeout = 30.0  # seconds
        self.max_connection_age = 3600.0  # 1 hour
        self.maintenance_interval = 300.0  # 5 minutes
        self.enable_wal_mode = True
        self.enable_foreign_keys = True
        self.busy_timeout = 30000  # 30 seconds
        
        logger.info(f"Created enhanced database pool: {self.db_path} (size: {pool_size})")
    
    async def initialize(self):
        """Initialize the database pool with comprehensive setup"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Ensure database directory exists
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create initial connection for setup
                async with aiosqlite.connect(
                    self.db_path,
                    timeout=self.connection_timeout
                ) as setup_conn:
                    # Configure database settings
                    await self._configure_connection(setup_conn)
                    
                    # Initialize schema
                    await init_schema(setup_conn)
                    
                    # Test the database
                    await setup_conn.execute("SELECT 1")
                
                # Pre-populate the pool
                for i in range(min(2, self.pool_size)):  # Start with 2 connections
                    try:
                        conn = await self._create_connection()
                        self._pool.append(conn)
                        self._connection_count += 1
                        logger.debug(f"Pre-created connection {i+1}")
                    except Exception as e:
                        logger.warning(f"Failed to pre-create connection {i+1}: {e}")
                
                self._initialized = True
                self._last_maintenance = datetime.now(UTC)
                
                # Start maintenance task
                asyncio.create_task(self._maintenance_task())
                
                logger.info(f"Initialized database pool with {len(self._pool)} connections")
                
                # Record initialization metric
                await add_custom_metric(
                    name="database.pool.initialized",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags={"db_path": str(self.db_path)}
                )
                
            except Exception as e:
                error = DatabaseException(
                    f"Failed to initialize database pool: {str(e)}",
                    error_code=ErrorCode.DB_CONNECTION_FAILED,
                    context=ErrorContext(
                        operation="pool_initialization",
                        component="enhanced_async_database_pool",
                        additional_data={"db_path": str(self.db_path)}
                    ),
                    original_exception=e
                )
                await error_tracker.track_error(error)
                raise error
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection with proper configuration"""
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            
            await self._configure_connection(conn)
            
            # Test the connection
            await conn.execute("SELECT 1")
            
            logger.debug(f"Created new database connection: {id(conn)}")
            return conn
            
        except Exception as e:
            error = DatabaseException(
                f"Failed to create database connection: {str(e)}",
                error_code=ErrorCode.DB_CONNECTION_FAILED,
                context=ErrorContext(
                    operation="connection_creation",
                    component="enhanced_async_database_pool"
                ),
                original_exception=e
            )
            await error_tracker.track_error(error)
            raise error
    
    async def _configure_connection(self, conn: aiosqlite.Connection):
        """Configure connection with optimal settings"""
        try:
            # Set busy timeout
            await conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout}")
            
            # Enable WAL mode for better concurrency
            if self.enable_wal_mode:
                await conn.execute("PRAGMA journal_mode = WAL")
            
            # Enable foreign key constraints
            if self.enable_foreign_keys:
                await conn.execute("PRAGMA foreign_keys = ON")
            
            # Set other performance settings
            await conn.execute("PRAGMA synchronous = NORMAL")  # Balance between safety and speed
            await conn.execute("PRAGMA cache_size = -64000")   # 64MB cache
            await conn.execute("PRAGMA temp_store = MEMORY")   # Store temp tables in memory
            await conn.execute("PRAGMA mmap_size = 268435456") # 256MB memory map
            
            # Connection-specific settings
            conn.isolation_level = None  # Autocommit mode
            
        except Exception as e:
            logger.warning(f"Failed to configure connection: {e}")
            # Don't raise here as the connection might still be usable
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool with comprehensive error handling"""
        if not self._initialized:
            await self.initialize()
        
        if self._closed:
            raise DatabaseException(
                "Database pool is closed",
                error_code=ErrorCode.DB_CONNECTION_FAILED
            )
        
        # Wait for available connection slot
        acquisition_start = time.time()
        
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.connection_timeout
            )
        except asyncio.TimeoutError:
            self._connection_timeouts += 1
            await add_custom_metric(
                name="database.pool.acquisition_timeout",
                value=1,
                metric_type=MetricType.COUNTER
            )
            raise DatabaseException(
                f"Connection acquisition timed out after {self.connection_timeout}s",
                error_code=ErrorCode.DB_TIMEOUT,
                context=ErrorContext(
                    operation="connection_acquisition",
                    component="enhanced_async_database_pool",
                    additional_data={
                        "timeout": self.connection_timeout,
                        "active_connections": self._active_connections,
                        "pool_size": self.pool_size
                    }
                )
            )
        
        conn: Optional[aiosqlite.Connection] = None
        connection_created = False
        
        try:
            async with self._lock:
                # Try to get connection from pool
                if self._pool:
                    conn = self._pool.pop()
                    logger.debug(f"Acquired connection from pool: {id(conn)}")
                else:
                    # Create new connection if pool is empty
                    if self._connection_count < self.pool_size:
                        conn = await self._create_connection()
                        self._connection_count += 1
                        connection_created = True
                        logger.debug(f"Created new connection: {id(conn)}")
                    else:
                        # This shouldn't happen with semaphore, but handle gracefully
                        raise DatabaseException(
                            "Pool exhausted and cannot create more connections",
                            error_code=ErrorCode.DB_CONNECTION_FAILED
                        )
                
                self._active_connections += 1
                self._total_acquisitions += 1
            
            # Validate connection health
            if not await self._is_connection_healthy(conn):
                logger.warning(f"Unhealthy connection detected: {id(conn)}")
                await self._close_connection(conn)
                
                # Create replacement connection
                async with self._lock:
                    conn = await self._create_connection()
                    if not connection_created:
                        self._connection_count += 1
                    logger.debug(f"Replaced unhealthy connection: {id(conn)}")
            
            acquisition_duration = (time.time() - acquisition_start) * 1000
            await add_custom_metric(
                name="database.pool.acquisition_duration_ms",
                value=acquisition_duration,
                metric_type=MetricType.TIMER
            )
            
            logger.debug(f"Connection acquired in {acquisition_duration:.1f}ms")
            
            yield conn
            
        except Exception as e:
            self._failed_acquisitions += 1
            
            if conn and not connection_created:
                # Return connection to pool if it was from pool
                async with self._lock:
                    if not self._closed and len(self._pool) < self.pool_size:
                        self._pool.append(conn)
                    else:
                        await self._close_connection(conn)
                        self._connection_count -= 1
                    self._active_connections = max(0, self._active_connections - 1)
            
            await add_custom_metric(
                name="database.pool.acquisition_error",
                value=1,
                metric_type=MetricType.COUNTER,
                tags={"error_type": type(e).__name__}
            )
            
            # Convert to database exception if needed
            if not isinstance(e, BaseRedTeamException):
                error = DatabaseException(
                    f"Connection acquisition failed: {str(e)}",
                    error_code=ErrorCode.DB_CONNECTION_FAILED,
                    original_exception=e
                )
                await error_tracker.track_error(error)
                raise error
            else:
                raise
        
        finally:
            # Always return connection to pool or close it
            if conn:
                try:
                    # Test if connection is still valid before returning to pool
                    await conn.execute("SELECT 1")
                    
                    async with self._lock:
                        if not self._closed and len(self._pool) < self.pool_size:
                            self._pool.append(conn)
                            logger.debug(f"Returned connection to pool: {id(conn)}")
                        else:
                            # Pool full or closed, close excess connection
                            await self._close_connection(conn)
                            self._connection_count -= 1
                            logger.debug(f"Closed excess connection: {id(conn)}")
                        
                        self._active_connections = max(0, self._active_connections - 1)
                
                except Exception as e:
                    # Connection is bad, just close it
                    logger.warning(f"Connection test failed, closing: {e}")
                    await self._close_connection(conn)
                    async with self._lock:
                        self._connection_count = max(0, self._connection_count - 1)
                        self._active_connections = max(0, self._active_connections - 1)
            
            # Release semaphore
            self._semaphore.release()
    
    async def _is_connection_healthy(self, conn: aiosqlite.Connection) -> bool:
        """Check if a connection is healthy"""
        try:
            # Simple connectivity test
            await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5.0)
            return True
        except Exception:
            return False
    
    async def _close_connection(self, conn: aiosqlite.Connection):
        """Safely close a connection"""
        try:
            await conn.close()
            logger.debug(f"Closed connection: {id(conn)}")
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
    
    async def _maintenance_task(self):
        """Background maintenance task for connection pool"""
        while not self._closed:
            try:
                await asyncio.sleep(self.maintenance_interval)
                
                if self._closed:
                    break
                
                await self._perform_maintenance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
                await asyncio.sleep(60)  # Backoff on error
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance on the connection pool"""
        if self._closed:
            return
        
        async with self._lock:
            current_time = datetime.now(UTC)
            self._last_maintenance = current_time
            
            # Check pool health
            healthy_connections = []
            for conn in self._pool:
                if await self._is_connection_healthy(conn):
                    healthy_connections.append(conn)
                else:
                    await self._close_connection(conn)
                    self._connection_count -= 1
                    logger.info("Removed unhealthy connection during maintenance")
            
            self._pool = healthy_connections
            
            # Record maintenance metrics
            await add_custom_metric(
                name="database.pool.size",
                value=len(self._pool),
                metric_type=MetricType.GAUGE
            )
            
            await add_custom_metric(
                name="database.pool.active_connections",
                value=self._active_connections,
                metric_type=MetricType.GAUGE
            )
            
            logger.debug(f"Pool maintenance completed: {len(self._pool)} healthy connections")
    
    async def close(self):
        """Close all connections in the pool"""
        if self._closed:
            return
        
        async with self._lock:
            self._closed = True
            
            for conn in self._pool:
                try:
                    await conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._pool.clear()
            self._connection_count = 0
            self._active_connections = 0
            
            logger.info("Database pool closed")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        async with self._lock:
            return {
                "pool_size": self.pool_size,
                "available_connections": len(self._pool),
                "active_connections": self._active_connections,
                "total_connections": self._connection_count,
                "total_acquisitions": self._total_acquisitions,
                "failed_acquisitions": self._failed_acquisitions,
                "connection_timeouts": self._connection_timeouts,
                "last_maintenance": self._last_maintenance.isoformat() if self._last_maintenance else None,
                "initialized": self._initialized,
                "closed": self._closed,
                "configuration": {
                    "connection_timeout": self.connection_timeout,
                    "max_connection_age": self.max_connection_age,
                    "maintenance_interval": self.maintenance_interval,
                    "enable_wal_mode": self.enable_wal_mode,
                    "enable_foreign_keys": self.enable_foreign_keys,
                    "busy_timeout": self.busy_timeout
                }
            }


# Backwards compatibility
AsyncDatabasePool = EnhancedAsyncDatabasePool
        
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