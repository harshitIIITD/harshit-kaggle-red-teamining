# ABOUTME: Test suite for UI dashboard endpoints and HTML rendering functionality
# ABOUTME: Validates /ui and /status endpoints return expected structures and content

import pytest
from fastapi.testclient import TestClient
from apps.runner.app.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app with lifespan management."""
    with TestClient(app) as client:
        yield client


def test_ui_contains_required_sections(client):
    """Test that /ui endpoint returns HTML with required dashboard sections."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    
    html_content = response.text
    
    # Check for required sections
    assert "Dashboard" in html_content
    assert "Total Attempts" in html_content
    assert "Total Cost" in html_content
    assert "Findings" in html_content
    assert "Status" in html_content
    
    # Check for auto-refresh meta tag
    assert '<meta http-equiv="refresh"' in html_content
    
    # Check for basic styling
    assert "<style>" in html_content or 'style="' in html_content


def test_status_json_counts(client):
    """Test that /status endpoint returns JSON with expected counter fields."""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    
    data = response.json()
    
    # Check required fields exist
    required_fields = [
        "status",
        "total_attempts",
        "successful_attempts",
        "failed_attempts", 
        "total_cost",
        "findings_count",
        "categories",
        "current_run_id",
        "error_rate"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Check data types
    assert isinstance(data["status"], str)
    assert isinstance(data["total_attempts"], int)
    assert isinstance(data["successful_attempts"], int)
    assert isinstance(data["failed_attempts"], int)
    assert isinstance(data["total_cost"], (int, float))
    assert isinstance(data["findings_count"], int)
    assert isinstance(data["categories"], dict)
    
    # Check non-negative values
    assert data["total_attempts"] >= 0
    assert data["successful_attempts"] >= 0
    assert data["failed_attempts"] >= 0
    assert data["total_cost"] >= 0
    assert data["findings_count"] >= 0
    assert data["error_rate"] >= 0.0
    
    # Check consistency
    assert data["total_attempts"] == data["successful_attempts"] + data["failed_attempts"]


def test_ui_links_to_report(client):
    """Test that UI contains link to current report if available."""
    response = client.get("/ui")
    assert response.status_code == 200
    
    html_content = response.text
    
    # Should have placeholder or link for report
    assert "Report" in html_content or "report" in html_content


def test_ui_shows_run_state(client):
    """Test that UI displays current run state (idle/running/paused)."""
    response = client.get("/ui")
    assert response.status_code == 200
    
    html_content = response.text
    
    # Should display one of the states
    assert any(state in html_content.lower() for state in ["idle", "running", "paused", "stopped"])


def test_status_includes_timestamp(client):
    """Test that status includes a timestamp field."""
    response = client.get("/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "timestamp" in data
    
    # Timestamp should be an ISO format string
    from datetime import datetime
    try:
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pytest.fail("Timestamp is not in valid ISO format")


def test_ui_responsive_design(client):
    """Test that UI uses responsive design elements."""
    response = client.get("/ui")
    assert response.status_code == 200
    
    html_content = response.text
    
    # Check for viewport meta tag
    assert '<meta name="viewport"' in html_content
    
    # Check for basic responsive CSS or inline styles
    assert any(term in html_content for term in ["max-width", "flex", "grid", "responsive"])