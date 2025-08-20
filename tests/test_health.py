# ABOUTME: Tests for health check endpoints (/status, /ui, /control)
# ABOUTME: Validates that the FastAPI app responds correctly to basic health checks

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys


@pytest.fixture
def client(tmp_path):
    """Create test client with mocked config"""
    test_config = {
        "providers": {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "target_model": "openai/gpt-oss-20b",
                "judge_model": "meta-llama/llama-3.1-70b-instruct",
                "paraphrase_model": "meta-llama/llama-3.1-8b-instruct",
                "headers": {
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "gpt-oss-redteam",
                },
            }
        },
        "run": {
            "categories": ["jailbreak", "prompt_injection"],
            "max_concurrency": 8,
            "batch_size_per_category": 250,
            "explore_exploit": {
                "algo": "thompson",
                "explore_bias": 0.35,
            },
            "reasoning_levels": ["low", "medium", "high"],
            "harmony_format": True,
        },
        "evaluation": {
            "judge_threshold": 0.72,
            "novelty_threshold": 0.65,
            "cost_cap_usd": 10.0,
            "heuristics": {"enable": True},
        },
        "storage": {
            "sqlite_path": str(tmp_path / "state.db"),
            "transcripts_path": str(tmp_path / "attempts.jsonl"),
            "findings_path": str(tmp_path / "findings.jsonl"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "ui": {
            "refresh_seconds": 3,
        },
    }

    # Remove the module from cache if it exists
    if "apps.runner.app.main" in sys.modules:
        del sys.modules["apps.runner.app.main"]

    # Patch load_config before importing the app
    with patch("apps.runner.app.util.config.load_config") as mock_config:
        mock_config.return_value = test_config
        from apps.runner.app.main import app

        with TestClient(app) as test_client:
            yield test_client


def test_status_returns_json_counters(client):
    """Test that /status returns JSON with expected counters"""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "counters" in data
    assert "config" in data

    # Check counters structure
    counters = data["counters"]
    assert "total_attempts" in counters
    assert "successful_attempts" in counters
    assert "failed_attempts" in counters
    assert "findings_count" in counters
    assert "clusters_count" in counters
    assert "estimated_cost_usd" in counters

    # Check types
    assert isinstance(counters["total_attempts"], int)
    assert isinstance(counters["successful_attempts"], int)
    assert isinstance(counters["failed_attempts"], int)
    assert isinstance(counters["findings_count"], int)
    assert isinstance(counters["clusters_count"], int)
    assert isinstance(counters["estimated_cost_usd"], (int, float))


def test_ui_serves_basic_html(client):
    """Test that /ui returns HTML content"""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for basic HTML structure
    html = response.text
    assert "<html" in html.lower()
    assert "<body" in html.lower()
    assert "dashboard" in html.lower() or "status" in html.lower()


def test_control_pause_endpoint(client):
    """Test that /control/pause endpoint exists and responds"""
    response = client.post("/control/pause")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    # When idle, pause returns "not_running"
    assert data["status"] in ["paused", "pausing", "already_paused", "not_running"]

    # Test pausing when running (need to simulate running state)
    # This would be done with proper state management in integration tests


def test_control_resume_endpoint(client):
    """Test that /control/resume endpoint exists and responds"""
    response = client.post("/control/resume")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    # When idle, resume returns "cannot_resume"
    assert data["status"] in ["resumed", "resuming", "already_running", "cannot_resume"]

    # Test resuming when paused (need to simulate paused state)
    # This would be done with proper state management in integration tests


def test_control_stop_endpoint(client):
    """Test that /control/stop endpoint exists and responds"""
    response = client.post("/control/stop")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["stopped", "stopping"]


def test_root_redirects_to_ui(client):
    """Test that root path redirects to /ui"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [301, 302, 307, 308]
    assert "/ui" in response.headers.get("location", "")


def test_health_check_endpoint(client):
    """Test that /health endpoint returns OK"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data


def test_config_loaded_correctly(client):
    """Test that configuration is loaded and accessible"""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    config = data["config"]

    # Check key config sections exist
    assert "max_concurrency" in config
    assert "judge_threshold" in config
    assert "cost_cap_usd" in config
    assert config["max_concurrency"] == 8
    assert config["judge_threshold"] == 0.72
    assert config["cost_cap_usd"] == 10.0


def test_status_includes_run_state(client):
    """Test that /status includes current run state"""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert "run_state" in data
    assert data["run_state"] in ["idle", "running", "paused", "stopped", "error"]
