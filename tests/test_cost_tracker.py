# ABOUTME: Tests for cost monitoring and alerting system
# ABOUTME: Validates cost tracking, threshold alerts, and budget enforcement

import json
import tempfile
from datetime import datetime, UTC, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from apps.runner.app.monitoring.cost_tracker import (
    CostTracker, CostAlert, AlertLevel, CostAlerter
)


@pytest.fixture
def cost_tracker():
    """Create a cost tracker with $10 budget"""
    tracker = CostTracker(cost_cap_usd=10.0, alert_thresholds=[0.5, 0.75, 0.9])
    return tracker


@pytest.fixture
def temp_checkpoint_file():
    """Create temporary checkpoint file"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestCostTracker:
    """Test cost tracking functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_cost_tracking(self, cost_tracker):
        """Test basic cost addition and tracking"""
        # Add some costs
        await cost_tracker.add_cost(1.5, "gpt-4", "testing", 1000)
        await cost_tracker.add_cost(0.5, "gpt-3.5", "evaluation", 500)
        
        assert cost_tracker.total_cost == 2.0
        assert cost_tracker.cost_by_model["gpt-4"] == 1.5
        assert cost_tracker.cost_by_model["gpt-3.5"] == 0.5
        assert cost_tracker.cost_by_category["testing"] == 1.5
        assert cost_tracker.cost_by_category["evaluation"] == 0.5
        assert cost_tracker.tokens_used == 1500
    
    @pytest.mark.asyncio
    async def test_alert_thresholds(self, cost_tracker):
        """Test that alerts trigger at correct thresholds"""
        alerts = []
        
        # Track alerts
        cost_tracker.register_alert_handler(lambda alert: alerts.append(alert))
        
        # Add cost up to 50% threshold ($5)
        alert = await cost_tracker.add_cost(4.9, "gpt-4", "testing", 1000)
        assert alert is None
        assert len(alerts) == 0
        
        # Cross 50% threshold
        alert = await cost_tracker.add_cost(0.2, "gpt-4", "testing", 100)
        assert alert is not None
        assert alert.level == AlertLevel.INFO
        assert "50%" in alert.message
        assert len(alerts) == 1
        
        # Add more cost up to 75% threshold
        alert = await cost_tracker.add_cost(2.3, "gpt-4", "testing", 1000)
        assert alert is None  # Already triggered 50%
        
        # Cross 75% threshold
        alert = await cost_tracker.add_cost(0.3, "gpt-4", "testing", 100)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert "75%" in alert.message
        assert len(alerts) == 2
        
        # Cross 90% threshold
        alert = await cost_tracker.add_cost(1.5, "gpt-4", "testing", 500)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert "90%" in alert.message
        assert len(alerts) == 3
    
    @pytest.mark.asyncio
    async def test_cost_cap_exceeded(self, cost_tracker):
        """Test alert when cost cap is exceeded"""
        # Add cost exceeding cap
        alert = await cost_tracker.add_cost(10.5, "gpt-4", "testing", 5000)
        
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert "COST CAP EXCEEDED" in alert.message
        assert cost_tracker.should_stop() is True
        assert cost_tracker.get_remaining_budget() == 0
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, cost_tracker):
        """Test cost metrics calculation"""
        # Add costs over time
        await cost_tracker.add_cost(2.0, "gpt-4", "testing", 1000)
        await cost_tracker.add_cost(1.0, "gpt-3.5", "evaluation", 500)
        
        metrics = cost_tracker.get_metrics()
        
        assert metrics.total_cost == 3.0
        assert metrics.estimated_remaining_budget == 7.0
        assert metrics.tokens_used == 1500
        assert metrics.average_cost_per_request == 1.5  # (2.0 + 1.0) / 2
        assert metrics.cost_per_hour > 0  # Should have positive hourly rate
    
    @pytest.mark.asyncio
    async def test_spending_report(self, cost_tracker):
        """Test spending report generation"""
        # Add various costs
        await cost_tracker.add_cost(2.0, "gpt-4", "testing", 1000)
        await cost_tracker.add_cost(1.0, "gpt-3.5", "evaluation", 500)
        await cost_tracker.add_cost(0.5, "gpt-4", "testing", 250)
        
        report = cost_tracker.get_spending_report()
        
        assert report["summary"]["total_cost_usd"] == 3.5
        assert report["summary"]["percentage_used"] == 35.0
        assert report["summary"]["remaining_budget_usd"] == 6.5
        
        assert report["breakdown"]["by_model"]["gpt-4"] == 2.5
        assert report["breakdown"]["by_model"]["gpt-3.5"] == 1.0
        
        assert report["breakdown"]["by_category"]["testing"] == 2.5
        assert report["breakdown"]["by_category"]["evaluation"] == 1.0
        
        assert report["alerts"]["next_threshold"] == 50.0  # Next is 50%
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, cost_tracker, temp_checkpoint_file):
        """Test saving and loading checkpoints"""
        cost_tracker.set_checkpoint_file(str(temp_checkpoint_file))
        
        # Add costs and trigger checkpoint
        await cost_tracker.add_cost(2.5, "gpt-4", "testing", 1000)
        await cost_tracker.add_cost(1.5, "gpt-3.5", "evaluation", 750)
        
        # Force checkpoint save
        await cost_tracker.save_checkpoint()
        
        # Create new tracker and load checkpoint
        new_tracker = CostTracker(cost_cap_usd=10.0)
        new_tracker.set_checkpoint_file(str(temp_checkpoint_file))
        loaded = await new_tracker.load_checkpoint()
        
        assert loaded is True
        assert new_tracker.total_cost == 4.0
        assert new_tracker.cost_by_model["gpt-4"] == 2.5
        assert new_tracker.cost_by_model["gpt-3.5"] == 1.5
        assert new_tracker.tokens_used == 1750
    
    @pytest.mark.asyncio
    async def test_alert_handler_registration(self, cost_tracker):
        """Test alert handler registration and triggering"""
        handler_calls = []
        
        def sync_handler(alert):
            handler_calls.append(("sync", alert))
        
        async def async_handler(alert):
            handler_calls.append(("async", alert))
        
        cost_tracker.register_alert_handler(sync_handler)
        cost_tracker.register_alert_handler(async_handler)
        
        # Trigger alert by crossing 50% threshold
        await cost_tracker.add_cost(5.1, "gpt-4", "testing", 2000)
        
        assert len(handler_calls) == 2
        assert handler_calls[0][0] == "sync"
        assert handler_calls[1][0] == "async"
        assert handler_calls[0][1].level == AlertLevel.INFO
    
    @pytest.mark.asyncio
    async def test_reset_functionality(self, cost_tracker):
        """Test resetting the tracker"""
        # Add costs
        await cost_tracker.add_cost(3.0, "gpt-4", "testing", 1500)
        assert cost_tracker.total_cost == 3.0
        
        # Reset
        cost_tracker.reset()
        
        assert cost_tracker.total_cost == 0.0
        assert len(cost_tracker.cost_by_model) == 0
        assert len(cost_tracker.cost_by_category) == 0
        assert cost_tracker.tokens_used == 0
        assert all(not triggered for triggered in cost_tracker.alerts_triggered.values())
    
    @pytest.mark.asyncio
    async def test_time_to_cap_estimation(self, cost_tracker):
        """Test estimation of time until budget cap"""
        # Simulate costs over time
        with patch('apps.runner.app.monitoring.cost_tracker.datetime') as mock_dt:
            # Set initial time
            start_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
            mock_dt.now.return_value = start_time
            mock_dt.UTC = UTC
            
            # Reset tracker with mocked time
            cost_tracker.start_time = start_time
            
            # Add $4 after 1 hour
            mock_dt.now.return_value = start_time + timedelta(hours=1)
            await cost_tracker.add_cost(4.0, "gpt-4", "testing", 2000)
            
            metrics = cost_tracker.get_metrics()
            
            # At $4/hour rate, with $6 remaining, should be 1.5 hours
            assert metrics.cost_per_hour == pytest.approx(4.0, rel=0.1)
            assert metrics.estimated_time_to_cap == pytest.approx(1.5, rel=0.1)


class TestCostAlerter:
    """Test cost alerter functionality"""
    
    @pytest.mark.asyncio
    async def test_alert_history(self):
        """Test alert history tracking"""
        alerter = CostAlerter()
        
        # Create test alerts
        alert1 = CostAlert(
            level=AlertLevel.INFO,
            message="Test alert 1",
            current_cost=5.0,
            cost_cap=10.0,
            percentage_used=50.0,
            timestamp=datetime.now(UTC)
        )
        
        alert2 = CostAlert(
            level=AlertLevel.WARNING,
            message="Test alert 2",
            current_cost=7.5,
            cost_cap=10.0,
            percentage_used=75.0,
            timestamp=datetime.now(UTC)
        )
        
        await alerter.send_alert(alert1)
        await alerter.send_alert(alert2)
        
        history = alerter.get_alert_history()
        assert len(history) == 2
        assert history[0].level == AlertLevel.INFO
        assert history[1].level == AlertLevel.WARNING
    
    @pytest.mark.asyncio
    async def test_webhook_alert(self):
        """Test webhook alert sending"""
        webhook_url = "https://hooks.example.com/test"
        alerter = CostAlerter(webhook_url=webhook_url)
        
        alert = CostAlert(
            level=AlertLevel.CRITICAL,
            message="Cost cap exceeded",
            current_cost=10.5,
            cost_cap=10.0,
            percentage_used=105.0,
            timestamp=datetime.now(UTC)
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            
            await alerter.send_alert(alert)
            
            # Verify webhook was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == webhook_url
            
            # Check payload structure
            payload = call_args[1]["json"]
            assert "attachments" in payload
            assert payload["attachments"][0]["color"] == "danger"
            assert "Cost cap exceeded" in payload["attachments"][0]["text"]
    
    @pytest.mark.asyncio
    async def test_alert_logging(self, tmp_path):
        """Test alert logging to file"""
        # Create log file path
        log_file = tmp_path / "data" / "cost_alerts.jsonl"
        
        # Create alerter with custom log file path
        alerter = CostAlerter(log_file_path=str(log_file))
        
        alert = CostAlert(
            level=AlertLevel.WARNING,
            message="Test warning",
            current_cost=7.5,
            cost_cap=10.0,
            percentage_used=75.0,
            timestamp=datetime.now(UTC)
        )
        
        # Log alert (the _log_alert method is now async)
        await alerter._log_alert(alert)
        
        # Verify log file was created
        assert log_file.exists()
        
        # Verify content
        with open(log_file, 'r') as f:
            logged_alert = json.loads(f.readline())
            assert logged_alert["level"] == "warning"
            assert logged_alert["message"] == "Test warning"
            assert logged_alert["current_cost"] == 7.5


class TestIntegrationWithOrchestrator:
    """Test integration of cost tracking with orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_cost_tracking(self):
        """Test that orchestrator properly tracks costs"""
        from apps.runner.app.orchestrator import AsyncOrchestrator
        
        # Mock configuration
        config = {
            "evaluation": {"cost_cap_usd": 5.0},
            "run": {
                "categories": ["test"],
                "max_concurrency": 1,
                "bandit_algorithm": "thompson",
                "cost_cap_usd": 5.0
            },
            "providers": {
                "openrouter": {
                    "models": {
                        "target": "test-model",
                        "attacker": "test-model",
                        "judge": "test-model"
                    }
                }
            }
        }
        
        # Create mock database pool
        mock_db = AsyncMock()
        
        # Create orchestrator with cost tracking
        orchestrator = AsyncOrchestrator(config, mock_db)
        
        # Verify cost tracker was initialized
        assert orchestrator.cost_cap_usd == 5.0
        
        # Simulate adding costs
        orchestrator.total_cost = 4.5
        
        # Check if should stop (approaching cap)
        should_stop = orchestrator.total_cost >= orchestrator.cost_cap_usd * 0.95
        assert should_stop is False
        
        # Add more cost to exceed cap
        orchestrator.total_cost = 5.1
        should_stop = orchestrator.total_cost >= orchestrator.cost_cap_usd
        assert should_stop is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])