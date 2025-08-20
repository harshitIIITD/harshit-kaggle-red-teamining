# ABOUTME: Tests for SQLite database operations, schema creation, and WAL mode.
# ABOUTME: Verifies DAO classes and entity persistence with proper isolation.

import tempfile
import os
import pytest
from datetime import datetime, UTC

from apps.runner.app.store.db import (
    open_db, init_schema, RunDAO, TaskDAO, AttemptDAO, EvalDAO, FindingDAO
)
from apps.runner.app.util.schemas import RunStatus as RunState


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    wal_path = db_path + '-wal'
    shm_path = db_path + '-shm'
    if os.path.exists(wal_path):
        os.unlink(wal_path)
    if os.path.exists(shm_path):
        os.unlink(shm_path)


def test_create_schema(temp_db):
    """Test that schema creation works and creates expected tables."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    # Check tables exist
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    tables = {row[0] for row in cursor.fetchall()}
    
    expected_tables = {'runs', 'tasks', 'attempts', 'evaluations', 'findings', 'state'}
    assert expected_tables.issubset(tables), f"Missing tables. Found: {tables}"
    
    conn.close()


def test_wal_mode_on(temp_db):
    """Test that WAL mode is enabled."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    # Check journal mode
    cursor = conn.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.lower() == 'wal', f"Expected WAL mode, got {journal_mode}"
    
    conn.close()


def test_insert_run_task_attempt_roundtrip(temp_db):
    """Test inserting and retrieving runs, tasks, and attempts."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    # Create DAOs
    run_dao = RunDAO(conn)
    task_dao = TaskDAO(conn)
    attempt_dao = AttemptDAO(conn)
    
    # Insert a run
    run_id = run_dao.create_run(
        target_model="meta-llama/llama-3.1-8b-instruct",
        categories=["safety", "capability"],
        max_attempts=100
    )
    assert run_id is not None
    
    # Get the run
    run = run_dao.get_run(run_id)
    assert run is not None
    assert run['target_model'] == "meta-llama/llama-3.1-8b-instruct"
    assert run['state'] == RunState.PENDING.value
    
    # Update run state
    run_dao.update_run_state(run_id, RunState.RUNNING)
    run = run_dao.get_run(run_id)
    assert run['state'] == RunState.RUNNING.value
    
    # Create a task
    task_id = task_dao.create_task(
        run_id=run_id,
        category="safety",
        template_id="jailbreak_1",
        mutator_chain=["unicode", "persona"],
        seed=42
    )
    assert task_id is not None
    
    # Get the task
    task = task_dao.get_task(task_id)
    assert task is not None
    assert task['category'] == "safety"
    assert task['seed'] == 42
    
    # Create an attempt
    attempt_id = attempt_dao.create_attempt(
        task_id=task_id,
        prompt="Test prompt",
        model="meta-llama/llama-3.1-8b-instruct"
    )
    assert attempt_id is not None
    
    # Update attempt with response
    attempt_dao.update_attempt(
        attempt_id=attempt_id,
        response="Test response",
        usage={'prompt_tokens': 10, 'completion_tokens': 20},
        cost=0.001,
        error=None
    )
    
    # Get the attempt
    attempt = attempt_dao.get_attempt(attempt_id)
    assert attempt is not None
    assert attempt['response'] == "Test response"
    assert attempt['cost'] == 0.001
    
    # Test listing
    tasks = task_dao.list_tasks_by_run(run_id)
    assert len(tasks) == 1
    
    attempts = attempt_dao.list_attempts_by_task(task_id)
    assert len(attempts) == 1
    
    conn.close()


def test_evaluation_dao(temp_db):
    """Test evaluation DAO operations."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    run_dao = RunDAO(conn)
    task_dao = TaskDAO(conn)
    attempt_dao = AttemptDAO(conn)
    eval_dao = EvalDAO(conn)
    
    # Create prerequisite data
    run_id = run_dao.create_run("test-model", ["safety"], 10)
    task_id = task_dao.create_task(run_id, "safety", "template1", [], 1)
    attempt_id = attempt_dao.create_attempt(task_id, "prompt", "model")
    
    # Create evaluation
    eval_id = eval_dao.create_evaluation(
        attempt_id=attempt_id,
        is_vulnerable=True,
        severity="high",
        category="jailbreak",
        heuristic_score=0.8,
        judge_score=0.9,
        combined_score=0.85,
        rationale="Test rationale"
    )
    assert eval_id is not None
    
    # Get evaluation
    evaluation = eval_dao.get_evaluation_by_attempt(attempt_id)
    assert evaluation is not None
    assert evaluation['is_vulnerable'] == 1
    assert evaluation['severity'] == "high"
    assert evaluation['combined_score'] == 0.85
    
    conn.close()


def test_finding_dao(temp_db):
    """Test finding DAO operations."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    run_dao = RunDAO(conn)
    finding_dao = FindingDAO(conn)
    
    # Create prerequisite data
    run_id = run_dao.create_run("test-model", ["safety"], 10)
    
    # Create finding
    finding_id = finding_dao.create_finding(
        run_id=run_id,
        cluster_id="cluster_1",
        category="jailbreak",
        severity="high",
        title="Test Finding",
        description="Test description",
        evidence={"prompts": ["test1", "test2"]},
        novelty_score=0.95,
        representative_attempt_id=1
    )
    assert finding_id is not None
    
    # Get finding
    finding = finding_dao.get_finding(finding_id)
    assert finding is not None
    assert finding['title'] == "Test Finding"
    assert finding['novelty_score'] == 0.95
    
    # List findings
    findings = finding_dao.list_findings_by_run(run_id)
    assert len(findings) == 1
    
    # Update promotion
    finding_dao.update_promotion_status(finding_id, promoted=True)
    finding = finding_dao.get_finding(finding_id)
    assert finding['promoted'] == 1
    
    conn.close()


def test_state_persistence(temp_db):
    """Test state key-value persistence."""
    conn = open_db(temp_db)
    init_schema(conn)
    
    # Set state
    conn.execute("""
        INSERT OR REPLACE INTO state (key, value, updated_at)
        VALUES (?, ?, ?)
    """, ("RUN_STATE", "PAUSED", datetime.now(UTC).isoformat()))
    conn.commit()
    
    # Get state
    cursor = conn.execute("SELECT value FROM state WHERE key = ?", ("RUN_STATE",))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "PAUSED"
    
    # Update state
    conn.execute("""
        INSERT OR REPLACE INTO state (key, value, updated_at)
        VALUES (?, ?, ?)
    """, ("RUN_STATE", "RUNNING", datetime.now(UTC).isoformat()))
    conn.commit()
    
    cursor = conn.execute("SELECT value FROM state WHERE key = ?", ("RUN_STATE",))
    row = cursor.fetchone()
    assert row[0] == "RUNNING"
    
    conn.close()


def test_idempotent_schema_init(temp_db):
    """Test that schema initialization is idempotent."""
    conn = open_db(temp_db)
    
    # Initialize schema twice
    init_schema(conn)
    init_schema(conn)  # Should not raise
    
    # Verify tables still exist
    cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]
    assert table_count >= 6  # At least our expected tables
    
    conn.close()


def test_concurrent_access(temp_db):
    """Test that WAL mode allows concurrent readers."""
    conn1 = open_db(temp_db)
    init_schema(conn1)
    
    # Insert data with first connection
    run_dao1 = RunDAO(conn1)
    run_id = run_dao1.create_run("model1", ["safety"], 10)
    
    # Open second connection for reading
    conn2 = open_db(temp_db)
    run_dao2 = RunDAO(conn2)
    
    # Should be able to read while first connection is open
    run = run_dao2.get_run(run_id)
    assert run is not None
    assert run['target_model'] == "model1"
    
    # Both connections should work
    run_dao1.update_run_state(run_id, RunState.COMPLETED)
    
    # Second connection should see the update (after its next transaction)
    conn2.execute("BEGIN")
    conn2.execute("COMMIT")
    run = run_dao2.get_run(run_id)
    assert run['state'] == RunState.COMPLETED.value
    
    conn1.close()
    conn2.close()