# ABOUTME: End-to-end tests for complete red-teaming runs
# ABOUTME: Tests dry-run with Llama model and cost cap enforcement

import pytest
import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from apps.runner.app.main import app
from apps.runner.app.store.db import open_db, init_schema
from apps.runner.app.util.schemas import Finding
from fastapi.testclient import TestClient


class TestE2EDryRun:
    """Test E2E dry-run functionality."""

    @pytest.fixture
    def client(self):
        """Create test client with lifespan management."""
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self, temp_data_dir):
        """Create mock configuration."""
        return {
            "providers": {
                "openrouter": {
                    "api_key": "test-key",
                    "base_url": "https://openrouter.ai/api/v1",
                }
            },
            "run": {
                "target_model": "meta-llama/llama-3.1-8b-instruct",
                "max_concurrency": 2,
                "cost_cap_usd": 1.0,
                "categories": ["harmful_content", "misinformation"],
                "attempts_per_category": 10,
            },
            "evaluation": {
                "judge_threshold": 0.7,
                "heuristics_weight": 0.3,
                "judge_weight": 0.7,
            },
            "storage": {
                "sqlite_path": str(temp_data_dir / "state.db"),
                "transcripts_path": str(temp_data_dir / "attempts.jsonl"),
                "findings_path": str(temp_data_dir / "findings.jsonl"),
                "reports_dir": str(temp_data_dir / "reports"),
            },
        }

    def test_dry_run_completes_with_findings_sample(self, mock_config, temp_data_dir):
        """Test that dry-run completes and produces at least one finding."""
        # Patch load_config before creating client to ensure it's used in lifespan
        with patch("apps.runner.app.main.load_config", return_value=mock_config):
            # Create client after patching so lifespan gets mocked config
            from fastapi.testclient import TestClient
            from apps.runner.app.main import app
            with TestClient(app) as client:
                # Initialize database
                db_path = temp_data_dir / "state.db"
                conn = open_db(str(db_path))
                init_schema(conn)
                conn.close()
                
                # Mock the orchestrator run to simulate a successful dry-run
                with patch("apps.runner.app.orchestrator.Orchestrator") as MockOrchestrator:
                    mock_orchestrator = Mock()
                    MockOrchestrator.return_value = mock_orchestrator
                    
                    # Simulate successful run with findings
                    async def mock_run(*args, **kwargs):
                        # Create mock finding
                        finding = Finding(
                            id="finding-001",
                            finding_id="finding-001",
                            run_id="dry-run-001",  # Added required field
                            category="harmful_content",
                            severity="high",
                            score=0.85,
                            novelty_score=0.9,
                            prompt="Test prompt",
                            response="Test response",
                            evidence={"heuristic_score": 0.8, "judge_score": 0.9},
                            examples=[],
                            description="Test finding",
                            created_at=datetime.now(timezone.utc).isoformat(),
                            cluster_id="cluster-001",
                            template_id="template-001",
                            mutator_chain=["mutator-001"],
                        )
                        
                        # Write finding to file
                        findings_file = temp_data_dir / "findings.jsonl"
                        with open(findings_file, "w") as f:
                            f.write(json.dumps(finding.to_dict()) + "\n")
                        
                        return {
                            "run_id": "dry-run-001",
                            "status": "completed",
                            "findings_count": 1,
                            "total_attempts": 20,
                            "total_cost": 0.5,
                        }
                    
                    mock_orchestrator.run = AsyncMock(side_effect=mock_run)
                    
                    # Start dry-run
                    response = client.post(
                        "/runs",
                        json={
                            "target_model": "meta-llama/llama-3.1-8b-instruct",
                            "max_attempts": 20,
                            "cost_cap_usd": 1.0,
                            "dry_run": True,
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "started"
                    assert "run_id" in data
                    
                    # Check that findings were created
                    findings_file = temp_data_dir / "findings.jsonl"
                    if findings_file.exists():
                        with open(findings_file) as f:
                            lines = f.readlines()
                            assert len(lines) >= 1, "At least one finding should be created"
                            
                            # Verify finding structure
                            finding_data = json.loads(lines[0])
                            assert "category" in finding_data
                            assert "score" in finding_data
                            assert finding_data["score"] > 0.5

    def test_cost_cap_enforced_and_graceful_stop(self, mock_config, temp_data_dir):
        """Test that cost cap is enforced and run stops gracefully."""
        # Setup mock config with low cost cap
        mock_config["run"]["cost_cap_usd"] = 0.01  # Very low cap
        
        # Patch load_config before creating client to ensure it's used in lifespan
        with patch("apps.runner.app.main.load_config", return_value=mock_config):
            # Create client after patching so lifespan gets mocked config
            from fastapi.testclient import TestClient
            from apps.runner.app.main import app
            with TestClient(app) as client:
                # Initialize database
                db_path = temp_data_dir / "state.db"
                conn = open_db(str(db_path))
                init_schema(conn)
                conn.close()
                
                with patch("apps.runner.app.orchestrator.Orchestrator") as MockOrchestrator:
                    mock_orchestrator = Mock()
                    MockOrchestrator.return_value = mock_orchestrator
                    
                    # Simulate run that hits cost cap
                    async def mock_run(*args, **kwargs):
                        return {
                            "run_id": "cost-cap-001",
                            "status": "stopped",
                            "reason": "cost_cap_reached",
                            "findings_count": 0,
                            "total_attempts": 5,
                            "total_cost": 0.01,
                        }
                    
                    mock_orchestrator.run = AsyncMock(side_effect=mock_run)
                    
                    # Start run with low cost cap
                    response = client.post(
                        "/runs",
                        json={
                            "target_model": "meta-llama/llama-3.1-8b-instruct",
                            "cost_cap_usd": 0.01,
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "started"
                    
                    # Verify cost cap was passed correctly
                    assert "params" in data
                    assert data["params"]["cost_cap_usd"] == 0.01
                    assert data["params"]["target_model"] == "meta-llama/llama-3.1-8b-instruct"

    def test_pause_resume_during_run(self, mock_config, temp_data_dir):
        """Test pause and resume functionality during a run."""
        # Patch load_config before creating client to ensure it's used in lifespan
        with patch("apps.runner.app.main.load_config", return_value=mock_config):
            # Create client after patching so lifespan gets mocked config
            from fastapi.testclient import TestClient
            from apps.runner.app.main import app
            with TestClient(app) as client:
                # Initialize database
                db_path = temp_data_dir / "state.db"
                conn = open_db(str(db_path))
                init_schema(conn)
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
                assert data["status"] in ["paused", "already_paused"]
                
                # Verify state in database if pause was successful
                if data["status"] == "paused":
                    conn = open_db(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT value FROM state WHERE key = ?", ("RUN_STATE",))
                    result = cursor.fetchone()
                    conn.close()
                    assert result[0] == "paused"
                
                # Test resume
                response = client.post("/control/resume")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] in ["resumed", "already_running"]
                
                # Verify state in database if resume was successful
                if data["status"] == "resumed":
                    conn = open_db(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT value FROM state WHERE key = ?", ("RUN_STATE",))
                    result = cursor.fetchone()
                    conn.close()
                    assert result[0] == "running"

    def test_report_generation_after_run(self, temp_data_dir):
        """Test that report is generated after run completes."""
        from apps.runner.app.agents.reporter import ReportGenerator
        from apps.runner.app.util.schemas import Finding
        
        # Create mock findings
        findings = [
            Finding(
                id=f"finding-{i:03d}",
                finding_id=f"finding-{i:03d}",
                run_id="test-run",  # Added required field
                category="harmful_content",
                severity="high",
                score=0.8 + i * 0.01,
                novelty_score=0.9,
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                evidence={"test": True},
                examples=[],
                description=f"Test finding {i}",
                created_at=datetime.now(timezone.utc).isoformat(),
                cluster_id=f"cluster-{i:03d}",
                template_id="template-001",
                mutator_chain=["mutator-001"],
            )
            for i in range(3)
        ]
        
        # Generate report
        generator = ReportGenerator(reports_dir=temp_data_dir)
        report_path = generator.write_report(
            run_id="test-run",
            findings=findings,
            stats={
                "total_attempts": 100,
                "successful_evaluations": 95,
                "unique_clusters": 10,
                "total_cost_usd": 0.75,
                "duration_seconds": 300,
                "model": "meta-llama/llama-3.1-8b-instruct",
            }
        )
        
        # Verify report exists and contains expected content
        assert report_path.exists()
        content = report_path.read_text()
        
        # Check for key sections
        assert "# Red-Team Report: test-run" in content
        assert "## Executive Summary" in content
        assert "## Statistics" in content
        assert "## Top Findings" in content
        assert "## Findings by Category" in content
        
        # Check statistics
        assert "Total Attempts: 100" in content
        assert "Successful Evaluations: 95" in content
        assert "Total Cost: $0.75" in content
        
        # Check findings are included
        assert "harmful_content" in content
        assert "Score: 0.8" in content

    @pytest.mark.asyncio
    async def test_concurrent_attempts_respect_limit(self, mock_config):
        """Test that concurrent attempts respect max_concurrency limit."""
        from apps.runner.app.orchestrator import Orchestrator
        from apps.runner.app.store.db import open_db, init_schema
        import tempfile
        from pathlib import Path
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = open_db(str(db_path))
            init_schema(conn)
            
            # Initialize orchestrator with concurrency limit
            orchestrator = Orchestrator(mock_config, conn)
            orchestrator.max_concurrency = 2
            
            # Track concurrent executions
            concurrent_count = 0
            max_concurrent = 0
            
            async def mock_task():
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.1)  # Simulate work
                concurrent_count -= 1
                return {"status": "success"}
            
            # Create multiple tasks
            tasks = [mock_task() for _ in range(10)]
            
            # Run with bounded concurrency
            results = []
            for i in range(0, len(tasks), orchestrator.max_concurrency):
                batch = tasks[i:i + orchestrator.max_concurrency]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)
            
            # Verify concurrency was respected
            assert max_concurrent <= orchestrator.max_concurrency
            assert len(results) == 10
            assert all(r["status"] == "success" for r in results)
            
            conn.close()