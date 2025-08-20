# ABOUTME: SQLite database operations with WAL mode, schema management, and DAO classes.
# ABOUTME: Provides durable state storage for runs, tasks, attempts, evaluations, and findings.

import sqlite3
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

from apps.runner.app.util.schemas import RunStatus as RunState


def open_db(db_path: str) -> sqlite3.Connection:
    """Open database connection with WAL mode and optimized settings."""
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    # Enable WAL mode for concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA journal_size_limit=67108864")  # 64MB limit
    conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
    
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema. Idempotent - safe to run multiple times."""
    
    # Runs table
    conn.execute("""
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
    conn.execute("""
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
    conn.execute("""
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
    conn.execute("""
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
    
    # State table for persistent key-value storage
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Findings table (deduplicated, high-value discoveries)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            cluster_id TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            evidence TEXT,  -- JSON
            novelty_score REAL NOT NULL,
            representative_attempt_id INTEGER,
            promoted BOOLEAN DEFAULT FALSE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES runs(id),
            FOREIGN KEY (representative_attempt_id) REFERENCES attempts(id),
            UNIQUE(run_id, cluster_id)
        )
    """)
    
    # State table (key-value store for pause/resume)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_task_id ON attempts(task_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_attempt_id ON evaluations(attempt_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_promoted ON findings(promoted)")
    
    conn.commit()


class RunDAO:
    """Data Access Object for runs."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_run(self, target_model: str, categories: List[str], max_attempts: int, 
                   metadata: Optional[Dict] = None) -> int:
        """Create a new run and return its ID."""
        cursor = self.conn.execute("""
            INSERT INTO runs (target_model, categories, max_attempts, state, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            target_model,
            json.dumps(categories),
            max_attempts,
            RunState.PENDING.value,
            datetime.now(UTC).isoformat(),
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get run by ID."""
        cursor = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_run_state(self, run_id: int, state: RunState) -> None:
        """Update run state."""
        now = datetime.now(UTC).isoformat()
        if state == RunState.RUNNING:
            self.conn.execute("""
                UPDATE runs SET state = ?, started_at = ? WHERE id = ?
            """, (state.value, now, run_id))
        elif state in (RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED):
            self.conn.execute("""
                UPDATE runs SET state = ?, completed_at = ? WHERE id = ?
            """, (state.value, now, run_id))
        else:
            self.conn.execute("""
                UPDATE runs SET state = ? WHERE id = ?
            """, (state.value, run_id))
        self.conn.commit()
    
    def update_run_cost(self, run_id: int, cost_increment: float) -> None:
        """Increment run cost."""
        self.conn.execute("""
            UPDATE runs SET cost_so_far = cost_so_far + ? WHERE id = ?
        """, (cost_increment, run_id))
        self.conn.commit()
    
    def get_active_run(self) -> Optional[Dict[str, Any]]:
        """Get the currently active run (if any)."""
        cursor = self.conn.execute("""
            SELECT * FROM runs 
            WHERE state IN (?, ?) 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (RunState.RUNNING.value, RunState.PAUSED.value))
        row = cursor.fetchone()
        return dict(row) if row else None


class TaskDAO:
    """Data Access Object for tasks."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_task(self, run_id: int, category: str, template_id: str,
                    mutator_chain: List[str], seed: int) -> int:
        """Create a new task and return its ID."""
        cursor = self.conn.execute("""
            INSERT OR IGNORE INTO tasks 
            (run_id, category, template_id, mutator_chain, seed, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            category,
            template_id,
            json.dumps(mutator_chain),
            seed,
            'PENDING',
            datetime.now(UTC).isoformat()
        ))
        self.conn.commit()
        
        if cursor.lastrowid == 0:  # Task already exists
            cursor = self.conn.execute("""
                SELECT id FROM tasks 
                WHERE run_id = ? AND template_id = ? AND mutator_chain = ? AND seed = ?
            """, (run_id, template_id, json.dumps(mutator_chain), seed))
            return cursor.fetchone()[0]
        
        return cursor.lastrowid
    
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        cursor = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['mutator_chain'] = json.loads(result['mutator_chain'])
            return result
        return None
    
    def list_tasks_by_run(self, run_id: int, state: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks for a run, optionally filtered by state."""
        if state:
            cursor = self.conn.execute("""
                SELECT * FROM tasks WHERE run_id = ? AND state = ?
            """, (run_id, state))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM tasks WHERE run_id = ?
            """, (run_id,))
        
        tasks = []
        for row in cursor:
            task = dict(row)
            task['mutator_chain'] = json.loads(task['mutator_chain'])
            tasks.append(task)
        return tasks
    
    def update_task_state(self, task_id: int, state: str) -> None:
        """Update task state."""
        now = datetime.now(UTC).isoformat()
        if state == 'COMPLETED':
            self.conn.execute("""
                UPDATE tasks SET state = ?, completed_at = ? WHERE id = ?
            """, (state, now, task_id))
        else:
            self.conn.execute("""
                UPDATE tasks SET state = ? WHERE id = ?
            """, (state, task_id))
        self.conn.commit()
    
    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending tasks for processing."""
        cursor = self.conn.execute("""
            SELECT * FROM tasks 
            WHERE state = 'PENDING' 
            ORDER BY created_at 
            LIMIT ?
        """, (limit,))
        
        tasks = []
        for row in cursor:
            task = dict(row)
            task['mutator_chain'] = json.loads(task['mutator_chain'])
            tasks.append(task)
        return tasks


class AttemptDAO:
    """Data Access Object for attempts."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_attempt(self, task_id: int, prompt: str, model: str) -> int:
        """Create a new attempt and return its ID."""
        cursor = self.conn.execute("""
            INSERT INTO attempts (task_id, prompt, model, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            task_id,
            prompt,
            model,
            datetime.now(UTC).isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_attempt(self, attempt_id: int, response: Optional[str], 
                      usage: Optional[Dict], cost: Optional[float], 
                      error: Optional[str]) -> None:
        """Update attempt with response or error."""
        self.conn.execute("""
            UPDATE attempts 
            SET response = ?, usage = ?, cost = ?, error = ?, completed_at = ?
            WHERE id = ?
        """, (
            response,
            json.dumps(usage) if usage else None,
            cost,
            error,
            datetime.now(UTC).isoformat(),
            attempt_id
        ))
        self.conn.commit()
    
    def get_attempt(self, attempt_id: int) -> Optional[Dict[str, Any]]:
        """Get attempt by ID."""
        cursor = self.conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result['usage']:
                result['usage'] = json.loads(result['usage'])
            return result
        return None
    
    def list_attempts_by_task(self, task_id: int) -> List[Dict[str, Any]]:
        """List attempts for a task."""
        cursor = self.conn.execute("""
            SELECT * FROM attempts WHERE task_id = ? ORDER BY created_at
        """, (task_id,))
        
        attempts = []
        for row in cursor:
            attempt = dict(row)
            if attempt['usage']:
                attempt['usage'] = json.loads(attempt['usage'])
            attempts.append(attempt)
        return attempts


class EvalDAO:
    """Data Access Object for evaluations."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_evaluation(self, attempt_id: int, is_vulnerable: bool, 
                         severity: Optional[str], category: Optional[str],
                         heuristic_score: Optional[float], judge_score: Optional[float],
                         combined_score: float, rationale: Optional[str]) -> int:
        """Create evaluation for an attempt."""
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO evaluations 
            (attempt_id, is_vulnerable, severity, category, heuristic_score, 
             judge_score, combined_score, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt_id,
            is_vulnerable,
            severity,
            category,
            heuristic_score,
            judge_score,
            combined_score,
            rationale,
            datetime.now(UTC).isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_evaluation_by_attempt(self, attempt_id: int) -> Optional[Dict[str, Any]]:
        """Get evaluation for an attempt."""
        cursor = self.conn.execute("""
            SELECT * FROM evaluations WHERE attempt_id = ?
        """, (attempt_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


class FindingDAO:
    """Data Access Object for findings."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def create_finding(self, run_id: int, cluster_id: str, category: str,
                      severity: str, title: str, description: Optional[str],
                      evidence: Dict, novelty_score: float,
                      representative_attempt_id: Optional[int]) -> int:
        """Create a new finding."""
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO findings 
            (run_id, cluster_id, category, severity, title, description, 
             evidence, novelty_score, representative_attempt_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            cluster_id,
            category,
            severity,
            title,
            description,
            json.dumps(evidence),
            novelty_score,
            representative_attempt_id,
            datetime.now(UTC).isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_finding(self, finding_id: int) -> Optional[Dict[str, Any]]:
        """Get finding by ID."""
        cursor = self.conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['evidence'] = json.loads(result['evidence'])
            return result
        return None
    
    def list_findings_by_run(self, run_id: int, promoted_only: bool = False) -> List[Dict[str, Any]]:
        """List findings for a run."""
        if promoted_only:
            cursor = self.conn.execute("""
                SELECT * FROM findings 
                WHERE run_id = ? AND promoted = 1
                ORDER BY novelty_score DESC
            """, (run_id,))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM findings 
                WHERE run_id = ?
                ORDER BY novelty_score DESC
            """, (run_id,))
        
        findings = []
        for row in cursor:
            finding = dict(row)
            finding['evidence'] = json.loads(finding['evidence'])
            findings.append(finding)
        return findings
    
    def update_promotion_status(self, finding_id: int, promoted: bool) -> None:
        """Update finding promotion status."""
        self.conn.execute("""
            UPDATE findings SET promoted = ? WHERE id = ?
        """, (promoted, finding_id))
        self.conn.commit()