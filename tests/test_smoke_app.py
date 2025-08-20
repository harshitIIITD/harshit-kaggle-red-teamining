# ABOUTME: Smoke tests for end-to-end application integration
# ABOUTME: Verifies service startup, API endpoints, and basic workflow

import pytest
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from apps.runner.app.main import app
from apps.runner.app.store.db import open_db, init_schema


class TestSmokeApp:
    """Smoke tests for the application."""

    @pytest.fixture
    def client(self):
        """Create test client with lifespan management."""
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "providers": {
                    "openrouter": {
                        "api_key": "test-key",
                        "base_url": "https://openrouter.ai/api/v1",
                    }
                },
                "run": {
                    "target_model": "meta-llama/llama-3.1-8b-instruct",
                    "max_concurrency": 2,
                    "cost_cap_usd": 0.1,
                    "categories": ["safety"],
                    "bandit_algorithm": "thompson",
                },
                "evaluation": {
                    "judge_threshold": 0.7,
                    "heuristics_weight": 0.5,
                    "judge_weight": 0.5,
                },
                "storage": {
                    "sqlite_path": f"{tmpdir}/state.db",
                    "transcripts_path": f"{tmpdir}/attempts.jsonl",
                    "findings_path": f"{tmpdir}/findings.jsonl",
                    "reports_dir": f"{tmpdir}/reports",
                },
            }
            yield config

    def test_health_endpoint(self, client):
        """Test that health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_status_endpoint(self, client):
        """Test that status endpoint returns expected structure."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "run_state" in data
        assert "total_attempts" in data
        assert "successful_attempts" in data
        assert "failed_attempts" in data
        assert "total_cost" in data
        assert "findings_count" in data
        assert "categories" in data
        assert "error_rate" in data
        assert "timestamp" in data

    def test_ui_endpoint(self, client):
        """Test that UI dashboard is served."""
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Red-Teaming Dashboard" in response.text
        assert "System Status" in response.text
        assert "Controls" in response.text

    @patch("apps.runner.app.main.load_config")
    def test_pause_resume_workflow(self, mock_load_config, client, temp_config):
        """Test pause and resume control endpoints."""
        mock_load_config.return_value = temp_config
        
        # Initialize database
        db_path = temp_config["storage"]["sqlite_path"]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = open_db(db_path)
        init_schema(conn)
        
        # Set initial running state
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            ("RUN_STATE", "running")
        )
        conn.commit()
        conn.close()
        
        # Test pause
        response = client.post("/control/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["paused", "already_paused", "not_running"]
        
        # Verify state changed (if pause was successful)
        if data["status"] == "paused":
            conn = open_db(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM state WHERE key = ?", ("RUN_STATE",))
            result = cursor.fetchone()
            conn.close()
            assert result[0] == "paused"
        
        # Test resume
        response = client.post("/control/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["resumed", "already_running", "cannot_resume"]

    def test_runs_endpoint_dry_run(self, temp_config):
        """Test starting a dry-run via /runs endpoint."""
        # Patch load_config before creating client to ensure it's used in lifespan
        with patch("apps.runner.app.main.load_config", return_value=temp_config):
            # Create client after patching so lifespan gets mocked config
            from fastapi.testclient import TestClient
            from apps.runner.app.main import app
            with TestClient(app) as client:
                # Mock orchestrator inside the client context
                with patch("apps.runner.app.orchestrator.Orchestrator") as MockOrchestrator:
                    mock_orch = Mock()
                    MockOrchestrator.return_value = mock_orch
                    mock_orch.run = Mock(return_value={
                        "run_id": "test-run",
                        "status": "completed",
                        "total_attempts": 10,
                        "findings_count": 1,
                    })
                    
                    # Start dry-run
                    response = client.post(
                        "/runs",
                        json={
                            "dry_run": True,
                            "max_attempts": 10,
                            "cost_cap_usd": 0.1,
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "started"
                    assert "run_id" in data
                    assert data["params"]["dry_run"] is True
                    assert data["params"]["target_model"] == "meta-llama/llama-3.1-8b-instruct"

    def test_no_exceptions_on_normal_flow(self, client):
        """Test that normal API flow doesn't raise exceptions."""
        # This is a basic smoke test to ensure the app starts and responds
        try:
            # Hit main endpoints
            client.get("/health")
            client.get("/status")
            client.get("/ui")
            
            # Try control endpoints (they may fail gracefully)
            client.post("/control/pause")
            client.post("/control/resume")
            client.post("/control/stop")
            
        except Exception as e:
            pytest.fail(f"Unexpected exception during smoke test: {e}")

    @patch("apps.runner.app.main.load_config")
    def test_report_file_created(self, mock_load_config, temp_config):
        """Test that report file can be created."""
        from apps.runner.app.agents.reporter import ReportGenerator
        from apps.runner.app.util.schemas import Finding
        from datetime import datetime, timezone
        
        mock_load_config.return_value = temp_config
        
        # Create a finding
        finding = Finding(
            id="test-001",
            finding_id="test-001",
            run_id="test-run",
            category="safety",
            severity="high",
            score=0.9,
            novelty_score=0.8,
            prompt="Test prompt",
            response="Test response",
            evidence={},
            examples=[],
            description="Test finding",
            created_at=datetime.now(timezone.utc).isoformat(),
            cluster_id="cluster-001",
            template_id="template-001",
            mutator_chain=[],
        )
        
        # Generate report
        reports_dir = Path(temp_config["storage"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        generator = ReportGenerator(reports_dir=reports_dir)
        report_path = generator.write_report(
            run_id="test-run",
            findings=[finding],
            stats={"total_attempts": 10}
        )
        
        # Verify report exists
        assert report_path.exists()
        content = report_path.read_text()
        assert "Red-Team Report" in content
        assert "safety" in content.lower()

    def test_database_initialization(self, temp_config):
        """Test that database initializes correctly."""
        db_path = temp_config["storage"]["sqlite_path"]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        conn = open_db(db_path)
        init_schema(conn)
        
        # Check tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "runs" in tables
        assert "tasks" in tables
        assert "attempts" in tables
        assert "evaluations" in tables
        assert "findings" in tables
        assert "state" in tables
        
        conn.close()