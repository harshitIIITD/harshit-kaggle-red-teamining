# ABOUTME: Comprehensive tests for the async orchestrator implementation
# ABOUTME: Tests bandit algorithms, circuit breaker, task scheduling, and worker management

import asyncio
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid
import os

from apps.runner.app.orchestrator import (
    AsyncOrchestrator,
    CircuitBreaker,
    ThompsonSampling,
    TaskItem
)
from apps.runner.app.store.async_db import AsyncDatabasePool


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        pool = AsyncDatabasePool(str(db_path))
        await pool.initialize()
        
        # Initialize schema
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.commit()
        
        yield pool
        await pool.close()


@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        return {
            "run": {
                "max_concurrency": 2,
                "cost_cap_usd": 10.0,
                "categories": ["test_cat1", "test_cat2"],
                "target_model": "test-model",
                "max_attempts": 100
            },
            "storage": {
                "sqlite_path": str(Path(tmpdir) / "test.db"),
                "transcripts_path": str(Path(tmpdir) / "transcripts.jsonl"),
                "findings_path": str(Path(tmpdir) / "findings.jsonl"),
                "reports_dir": str(Path(tmpdir) / "reports")
            },
            "providers": {
                "openrouter": {
                    "base_url": "http://test",
                    "api_key": "test-key"
                }
            }
        }


class TestCircuitBreaker:
    """Test the circuit breaker functionality"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly"""
        cb = CircuitBreaker(threshold=0.5, window_size=10, cooldown_seconds=5)
        assert cb.threshold == 0.5
        assert cb.window_size == 10
        assert cb.cooldown_seconds == 5
        assert not cb.is_open()
    
    def test_circuit_breaker_trips_on_high_error_rate(self):
        """Test circuit breaker trips when error rate exceeds threshold"""
        cb = CircuitBreaker(threshold=0.5, window_size=4, cooldown_seconds=1)
        
        # Record failures to trip the breaker
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        # Should be tripped (75% failure rate > 50% threshold)
        assert cb.is_open()
    
    def test_circuit_breaker_stays_closed_on_low_error_rate(self):
        """Test circuit breaker stays closed when error rate is below threshold"""
        cb = CircuitBreaker(threshold=0.5, window_size=4, cooldown_seconds=1)
        
        # Record mostly successes
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        cb.record_success()
        
        # Should not be tripped (25% failure rate < 50% threshold)
        assert not cb.is_open()
    
    def test_circuit_breaker_cooldown(self):
        """Test circuit breaker resets after cooldown period"""
        import time
        cb = CircuitBreaker(threshold=0.5, window_size=2, cooldown_seconds=0.1)
        
        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Should be reset
        assert not cb.is_open()


class TestThompsonSampling:
    """Test the Thompson Sampling bandit algorithm"""
    
    def test_thompson_sampling_initialization(self):
        """Test Thompson Sampling initializes correctly"""
        ts = ThompsonSampling(alpha=1.0, beta=1.0)
        assert len(ts.arms) == 0
    
    def test_thompson_sampling_select_arm(self):
        """Test arm selection works correctly"""
        ts = ThompsonSampling()
        arms = ["arm1", "arm2", "arm3"]
        
        # Should select one of the arms
        selected = ts.select_arm(arms)
        assert selected in arms
    
    def test_thompson_sampling_update(self):
        """Test updating arm statistics"""
        ts = ThompsonSampling()
        
        # Update with success
        ts.update("arm1", reward=0.8)
        assert ts.arms["arm1"]["successes"] == 1
        assert ts.arms["arm1"]["failures"] == 0
        
        # Update with failure
        ts.update("arm1", reward=0.2)
        assert ts.arms["arm1"]["successes"] == 1
        assert ts.arms["arm1"]["failures"] == 1
    
    def test_thompson_sampling_exploration_vs_exploitation(self):
        """Test that Thompson Sampling balances exploration and exploitation"""
        ts = ThompsonSampling()
        
        # Give arm1 many successes
        for _ in range(10):
            ts.update("arm1", reward=0.9)
        
        # Give arm2 many failures
        for _ in range(10):
            ts.update("arm2", reward=0.1)
        
        # arm1 should be selected more often
        selections = {"arm1": 0, "arm2": 0}
        for _ in range(100):
            selected = ts.select_arm(["arm1", "arm2"])
            selections[selected] += 1
        
        # arm1 should be selected significantly more often
        assert selections["arm1"] > selections["arm2"]


class TestTaskItem:
    """Test the TaskItem dataclass"""
    
    def test_task_item_creation(self):
        """Test TaskItem creation and attributes"""
        task = TaskItem(
            id="test-1",
            category="test_cat",
            strategy="test_strat",
            template_id="template-1",
            priority=0.8,
            metadata={"key": "value"}
        )
        
        assert task.id == "test-1"
        assert task.category == "test_cat"
        assert task.strategy == "test_strat"
        assert task.priority == 0.8
    
    def test_task_item_priority_ordering(self):
        """Test TaskItem ordering by priority"""
        task1 = TaskItem("t1", "cat", "strat", "tmpl", 0.5, {})
        task2 = TaskItem("t2", "cat", "strat", "tmpl", 0.8, {})
        task3 = TaskItem("t3", "cat", "strat", "tmpl", 0.3, {})
        
        # Higher priority should be "less than" for max heap
        assert task2 < task1  # 0.8 > 0.5
        assert task1 < task3  # 0.5 > 0.3


@pytest.mark.asyncio
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
@patch("apps.runner.app.orchestrator.OpenRouterClient")
class TestAsyncOrchestrator:
    """Test the AsyncOrchestrator class"""
    
    async def test_orchestrator_initialization(self, mock_or_client, mock_config, temp_db):
        """Test orchestrator initializes correctly"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        
        assert orchestrator.max_concurrency == 2
        assert orchestrator.cost_cap_usd == 10.0
        assert orchestrator.categories == ["test_cat1", "test_cat2"]
        assert orchestrator.strategies == ["direct", "roleplay", "encoding"]
    
    async def test_orchestrator_generate_initial_tasks(self, mock_or_client, mock_config, temp_db):
        """Test initial task generation"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        await orchestrator.initialize()
        
        # Should generate tasks for each category/strategy combination
        # 2 categories * 3 strategies * 5 tasks each = 30 tasks
        task_count = orchestrator.task_queue.qsize()
        assert task_count == 30
    
    async def test_orchestrator_checkpoint(self, mock_or_client, mock_config, temp_db):
        """Test checkpoint saves state to database"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        orchestrator.total_cost = 5.5
        orchestrator.attempts_count = 10
        orchestrator.success_count = 8
        orchestrator.error_count = 2
        
        await orchestrator.checkpoint()
        
        # Verify state was saved
        async with temp_db.acquire() as conn:
            cursor = await conn.execute(
                "SELECT value FROM state WHERE key = 'TOTAL_COST'"
            )
            result = await cursor.fetchone()
            assert result[0] == "5.5"
    
    async def test_orchestrator_run_state_management(self, mock_or_client, mock_config, temp_db):
        """Test run state get/set operations"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        
        # Set state
        await orchestrator._set_run_state("running")
        
        # Get state
        state = await orchestrator._get_run_state()
        assert state == "running"
    
    async def test_orchestrator_task_execution_mock(self, mock_or_client, mock_config, temp_db):
        """Test task execution with mock"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        await orchestrator.initialize()  # Initialize to set transcript_file
        
        task = TaskItem(
            id="test-task-1",
            category="test_cat",
            strategy="test_strat",
            template_id="template-1",
            priority=0.8,
            metadata={}
        )
        
        result = await orchestrator._execute_task(task)
        
        assert result["task_id"] == "test-task-1"
        assert result["success"] == True
        assert "cost" in result
        assert "score" in result
    
    async def test_orchestrator_worker_respects_max_attempts(self, mock_or_client, mock_config, temp_db):
        """Test worker stops at max attempts"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        orchestrator.max_attempts = 2
        await orchestrator.initialize()
        
        # Run worker
        await orchestrator._task_worker(0)
        
        # Should have processed at most 2 attempts
        assert orchestrator.attempts_count <= 2
    
    async def test_orchestrator_worker_respects_cost_cap(self, mock_or_client, mock_config, temp_db):
        """Test worker stops when cost cap is reached"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        orchestrator.cost_cap_usd = 0.001  # Very low cap
        orchestrator.max_attempts = 100
        await orchestrator.initialize()
        
        # Run worker
        await orchestrator._task_worker(0)
        
        # Should have stopped due to cost cap
        assert orchestrator.total_cost <= 0.002  # Some tolerance for last task
    
    async def test_orchestrator_pause_resume(self, mock_or_client, mock_config, temp_db):
        """Test pause/resume functionality"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        await orchestrator.initialize()
        
        # Set paused state
        await orchestrator._set_run_state("paused")
        
        # Worker should not process tasks when paused
        initial_attempts = orchestrator.attempts_count
        
        # Run worker briefly
        worker_task = asyncio.create_task(orchestrator._task_worker(0))
        await asyncio.sleep(0.1)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        # No new attempts should have been made
        assert orchestrator.attempts_count == initial_attempts
    
    async def test_orchestrator_circuit_breaker_integration(self, mock_or_client, mock_config, temp_db):
        """Test circuit breaker pauses execution on high error rate"""
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        
        # Force circuit breaker to trip
        for _ in range(10):
            orchestrator.circuit_breaker.record_failure()
        
        assert orchestrator.circuit_breaker.is_open()
        
        # Worker should pause when circuit breaker is open
        await orchestrator.initialize()
        initial_attempts = orchestrator.attempts_count
        
        # Run worker briefly
        worker_task = asyncio.create_task(orchestrator._task_worker(0))
        await asyncio.sleep(0.1)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        
        # Should have paused due to circuit breaker
        state = await orchestrator._get_run_state()
        assert state == "paused"
    
    async def test_orchestrator_run_complete_flow(self, mock_or_client, mock_config, temp_db):
        """Test complete orchestrator run flow"""
        # Configure the mock OpenRouterClient
        mock_instance = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_or_client.return_value = mock_instance
        
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        
        # Mock detector and reporter
        mock_detector = Mock()
        mock_detector.get_findings.return_value = []
        mock_detector.cluster_store = Mock()
        mock_detector.cluster_store.clusters = []
        
        mock_reporter = Mock()
        
        # Run with very limited attempts
        result = await orchestrator.run(
            run_id="test-run-1",
            max_attempts=5,
            detector=mock_detector,
            reporter=mock_reporter
        )
        
        assert result["run_id"] == "test-run-1"
        assert result["status"] == "completed"
        assert result["total_attempts"] <= 6  # Allow for slight timing variation
        assert "bandit_stats" in result
        assert "duration_seconds" in result
    
    async def test_orchestrator_concurrent_workers(self, mock_or_client, mock_config, temp_db):
        """Test multiple workers run concurrently"""
        mock_config["run"]["max_concurrency"] = 3
        orchestrator = AsyncOrchestrator(mock_config, temp_db)
        orchestrator.max_attempts = 10
        await orchestrator.initialize()
        
        # Track which workers executed
        worker_executed = set()
        
        async def tracked_worker(worker_id):
            worker_executed.add(worker_id)
            await orchestrator._task_worker(worker_id)
        
        # Run multiple workers
        workers = [
            asyncio.create_task(tracked_worker(i))
            for i in range(3)
        ]
        
        # Let them run briefly
        await asyncio.sleep(0.5)
        
        # Cancel all workers
        for w in workers:
            w.cancel()
        
        for w in workers:
            try:
                await w
            except asyncio.CancelledError:
                pass
        
        # All workers should have started
        assert len(worker_executed) == 3