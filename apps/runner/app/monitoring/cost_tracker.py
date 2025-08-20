# ABOUTME: Cost monitoring and alerting system for tracking API spending
# ABOUTME: Provides real-time cost tracking, budget alerts, and spending reports

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import aiofiles

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostAlert:
    """Cost alert data structure"""
    level: AlertLevel
    message: str
    current_cost: float
    cost_cap: float
    percentage_used: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "level": self.level.value,
            "message": self.message,
            "current_cost": self.current_cost,
            "cost_cap": self.cost_cap,
            "percentage_used": self.percentage_used,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CostMetrics:
    """Cost metrics for monitoring"""
    total_cost: float
    cost_by_model: Dict[str, float]
    cost_by_category: Dict[str, float]
    cost_per_hour: float
    estimated_remaining_budget: float
    estimated_time_to_cap: Optional[float]  # Hours until budget exhausted
    tokens_used: int
    average_cost_per_request: float
    

class CostTracker:
    """Tracks API costs and provides alerts when approaching limits"""
    
    def __init__(self, cost_cap_usd: float, alert_thresholds: Optional[List[float]] = None):
        """
        Initialize cost tracker
        
        Args:
            cost_cap_usd: Maximum allowed cost in USD
            alert_thresholds: List of percentages at which to trigger alerts (default: [50, 75, 90, 95])
        """
        self.cost_cap_usd = cost_cap_usd
        self.alert_thresholds = alert_thresholds or [0.5, 0.75, 0.9, 0.95]
        
        # Tracking state
        self.total_cost = 0.0
        self.cost_by_model: Dict[str, float] = {}
        self.cost_by_category: Dict[str, float] = {}
        # Use deque for efficient removal from left side
        self.request_costs: Deque[Tuple[datetime, float]] = deque(maxlen=10000)  # Limit size
        self.tokens_used = 0
        self.alerts_triggered: Dict[float, bool] = {t: False for t in self.alert_thresholds}
        
        # Start time for rate calculations
        self.start_time = datetime.now(UTC)
        self.last_checkpoint = datetime.now(UTC)
        
        # Alert callbacks
        self.alert_handlers: List[callable] = []
        
        # Persistence
        self.checkpoint_file: Optional[Path] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    def set_checkpoint_file(self, path: str) -> None:
        """Set file path for persisting cost data"""
        # Sanitize path to prevent directory traversal
        safe_path = os.path.normpath(path)
        if os.path.isabs(safe_path) and ".." in Path(safe_path).parts:
            raise ValueError(f"Invalid checkpoint path: {path}")
        
        self.checkpoint_file = Path(safe_path)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def add_cost(
        self,
        cost: float,
        model: str,
        category: str,
        tokens: int = 0
    ) -> Optional[CostAlert]:
        """
        Add a cost entry and check for alerts
        
        Args:
            cost: Cost in USD
            model: Model name that incurred the cost
            category: Category of request (e.g., "testing", "evaluation")
            tokens: Number of tokens used
            
        Returns:
            CostAlert if threshold crossed, None otherwise
        """
        # Validate input
        if cost < 0:
            raise ValueError(f"Cost cannot be negative: {cost}")
        if tokens < 0:
            raise ValueError(f"Tokens cannot be negative: {tokens}")
        
        async with self._lock:
            # Update totals
            self.total_cost += cost
            self.cost_by_model[model] = self.cost_by_model.get(model, 0) + cost
            self.cost_by_category[category] = self.cost_by_category.get(category, 0) + cost
            self.tokens_used += tokens
        
            # Track request for rate calculations
            now = datetime.now(UTC)
            self.request_costs.append((now, cost))
            
            # Clean old request data (keep last hour) - optimized
            cutoff = now - timedelta(hours=1)
            # Remove old entries from the left (oldest first)
            while self.request_costs and self.request_costs[0][0] <= cutoff:
                self.request_costs.popleft()
            
            # Check for alerts
            alert = self._check_alerts()
        
        # Trigger alert handlers (outside lock to avoid deadlock)
        if alert:
            await self._trigger_alert(alert)
        
        # Save checkpoint
        if self.checkpoint_file and (datetime.now(UTC) - self.last_checkpoint).total_seconds() > 60:
            await self.save_checkpoint()
            self.last_checkpoint = datetime.now(UTC)
        
        return alert
    
    def _check_alerts(self) -> Optional[CostAlert]:
        """Check if any alert thresholds have been crossed"""
        percentage_used = (self.total_cost / self.cost_cap_usd) * 100
        
        # Check if cost cap exceeded
        if self.total_cost >= self.cost_cap_usd:
            return CostAlert(
                level=AlertLevel.CRITICAL,
                message=f"COST CAP EXCEEDED: ${self.total_cost:.2f} >= ${self.cost_cap_usd:.2f}",
                current_cost=self.total_cost,
                cost_cap=self.cost_cap_usd,
                percentage_used=percentage_used,
                timestamp=datetime.now(UTC)
            )
        
        # Check threshold alerts
        for threshold in sorted(self.alert_thresholds, reverse=True):
            threshold_percent = threshold * 100
            if percentage_used >= threshold_percent and not self.alerts_triggered[threshold]:
                self.alerts_triggered[threshold] = True
                
                # Determine alert level
                if threshold >= 0.9:
                    level = AlertLevel.CRITICAL
                elif threshold >= 0.75:
                    level = AlertLevel.WARNING
                else:
                    level = AlertLevel.INFO
                
                return CostAlert(
                    level=level,
                    message=f"Cost threshold {threshold_percent:.0f}% reached: ${self.total_cost:.2f} / ${self.cost_cap_usd:.2f}",
                    current_cost=self.total_cost,
                    cost_cap=self.cost_cap_usd,
                    percentage_used=percentage_used,
                    timestamp=datetime.now(UTC)
                )
        
        return None
    
    async def _trigger_alert(self, alert: CostAlert) -> None:
        """Trigger registered alert handlers"""
        # Log the alert
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(alert.message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def register_alert_handler(self, handler: callable) -> None:
        """Register a callback for cost alerts"""
        self.alert_handlers.append(handler)
    
    def get_metrics(self) -> CostMetrics:
        """Get current cost metrics"""
        elapsed_hours = (datetime.now(UTC) - self.start_time).total_seconds() / 3600
        elapsed_hours = max(elapsed_hours, 0.001)  # Avoid division by zero
        
        # Calculate hourly rate
        cost_per_hour = self.total_cost / elapsed_hours
        
        # Estimate time to cap
        remaining_budget = self.cost_cap_usd - self.total_cost
        if cost_per_hour > 0 and remaining_budget > 0:
            estimated_time_to_cap = remaining_budget / cost_per_hour
        else:
            estimated_time_to_cap = None
        
        # Average cost per request  
        num_requests = len(self.request_costs)
        avg_cost = self.total_cost / num_requests if num_requests > 0 else 0
        
        return CostMetrics(
            total_cost=self.total_cost,
            cost_by_model=self.cost_by_model.copy(),
            cost_by_category=self.cost_by_category.copy(),
            cost_per_hour=cost_per_hour,
            estimated_remaining_budget=remaining_budget,
            estimated_time_to_cap=estimated_time_to_cap,
            tokens_used=self.tokens_used,
            average_cost_per_request=avg_cost
        )
    
    def get_spending_report(self) -> Dict:
        """Generate a detailed spending report"""
        metrics = self.get_metrics()
        
        return {
            "summary": {
                "total_cost_usd": round(self.total_cost, 4),
                "cost_cap_usd": self.cost_cap_usd,
                "percentage_used": round((self.total_cost / self.cost_cap_usd) * 100, 2),
                "remaining_budget_usd": round(metrics.estimated_remaining_budget, 4),
                "run_time_hours": round((datetime.now(UTC) - self.start_time).total_seconds() / 3600, 2)
            },
            "rates": {
                "cost_per_hour_usd": round(metrics.cost_per_hour, 4),
                "estimated_hours_to_cap": round(metrics.estimated_time_to_cap, 2) if metrics.estimated_time_to_cap else None,
                "average_cost_per_request_usd": round(metrics.average_cost_per_request, 6),
                "tokens_used": metrics.tokens_used
            },
            "breakdown": {
                "by_model": {k: round(v, 4) for k, v in metrics.cost_by_model.items()},
                "by_category": {k: round(v, 4) for k, v in metrics.cost_by_category.items()}
            },
            "alerts": {
                "triggered_thresholds": [t * 100 for t, triggered in self.alerts_triggered.items() if triggered],
                "next_threshold": self._get_next_threshold()
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    def _get_next_threshold(self) -> Optional[float]:
        """Get the next alert threshold percentage"""
        percentage_used = (self.total_cost / self.cost_cap_usd) * 100
        
        for threshold in sorted(self.alert_thresholds):
            if percentage_used < threshold * 100:
                return threshold * 100
        
        return None
    
    async def save_checkpoint(self) -> None:
        """Save current state to checkpoint file"""
        if not self.checkpoint_file:
            return
        
        checkpoint_data = {
            "total_cost": self.total_cost,
            "cost_by_model": self.cost_by_model,
            "cost_by_category": self.cost_by_category,
            "tokens_used": self.tokens_used,
            "alerts_triggered": {str(k): v for k, v in self.alerts_triggered.items()},
            "start_time": self.start_time.isoformat(),
            "last_checkpoint": datetime.now(UTC).isoformat()
        }
        
        try:
            async with aiofiles.open(self.checkpoint_file, 'w') as f:
                await f.write(json.dumps(checkpoint_data, indent=2))
        except PermissionError as e:
            logger.error(f"Permission denied when saving checkpoint to {self.checkpoint_file}: {e}")
        except OSError as e:
            logger.error(f"OS error when saving checkpoint: {e}")
        except Exception as e:
            logger.error(f"Failed to save cost checkpoint: {e}")
    
    async def load_checkpoint(self) -> bool:
        """Load state from checkpoint file"""
        if not self.checkpoint_file or not self.checkpoint_file.exists():
            return False
        
        try:
            async with aiofiles.open(self.checkpoint_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            self.total_cost = data.get("total_cost", 0.0)
            self.cost_by_model = data.get("cost_by_model", {})
            self.cost_by_category = data.get("cost_by_category", {})
            self.tokens_used = data.get("tokens_used", 0)
            
            # Restore alerts triggered state
            for k, v in data.get("alerts_triggered", {}).items():
                threshold = float(k)
                if threshold in self.alerts_triggered:
                    self.alerts_triggered[threshold] = v
            
            # Restore start time
            if "start_time" in data:
                self.start_time = datetime.fromisoformat(data["start_time"])
            
            logger.info(f"Loaded cost checkpoint: ${self.total_cost:.2f} spent")
            return True
        
        except PermissionError as e:
            logger.error(f"Permission denied when loading checkpoint from {self.checkpoint_file}: {e}")
            return False
        except OSError as e:
            logger.error(f"OS error when loading checkpoint: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load cost checkpoint: {e}")
            return False
    
    def should_stop(self) -> bool:
        """Check if execution should stop due to cost cap"""
        return self.total_cost >= self.cost_cap_usd
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget in USD"""
        return max(0, self.cost_cap_usd - self.total_cost)
    
    def reset(self) -> None:
        """Reset all tracking data"""
        self.total_cost = 0.0
        self.cost_by_model.clear()
        self.cost_by_category.clear()
        self.request_costs = deque(maxlen=10000)
        self.tokens_used = 0
        self.alerts_triggered = {t: False for t in self.alert_thresholds}
        self.start_time = datetime.now(UTC)
        self.last_checkpoint = datetime.now(UTC)


class CostAlerter:
    """Handles cost alert notifications"""
    
    def __init__(self, webhook_url: Optional[str] = None, email_config: Optional[Dict] = None,
                 log_file_path: Optional[str] = None):
        """
        Initialize alerter with notification channels
        
        Args:
            webhook_url: Optional webhook URL for alerts (e.g., Slack, Discord)
            email_config: Optional email configuration for alerts
            log_file_path: Optional custom path for alert logs (default: data/cost_alerts.jsonl)
        """
        self.webhook_url = self._validate_webhook_url(webhook_url) if webhook_url else None
        self.email_config = email_config
        self.alert_history: List[CostAlert] = []
        self.log_file_path = Path(log_file_path) if log_file_path else Path("data/cost_alerts.jsonl")
    
    def _validate_webhook_url(self, url: str) -> str:
        """Validate webhook URL format"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid webhook URL: {url}")
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(f"Webhook URL must use HTTP or HTTPS: {url}")
            return url
        except Exception as e:
            logger.error(f"Invalid webhook URL: {e}")
            raise ValueError(f"Invalid webhook URL: {url}")
    
    async def send_alert(self, alert: CostAlert) -> None:
        """Send alert through configured channels"""
        # Store in history
        self.alert_history.append(alert)
        
        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook(alert)
        
        # Send email if configured
        if self.email_config:
            await self._send_email(alert)
        
        # Always log locally
        await self._log_alert(alert)
    
    async def _send_webhook(self, alert: CostAlert) -> None:
        """Send alert to webhook"""
        if not self.webhook_url:
            return
            
        try:
            import httpx
            
            # Format message for webhook (Slack-compatible format)
            color = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning", 
                AlertLevel.CRITICAL: "danger"
            }.get(alert.level, "warning")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Cost Alert: {alert.level.value.upper()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Current Cost", "value": f"${alert.current_cost:.2f}", "short": True},
                        {"title": "Cost Cap", "value": f"${alert.cost_cap:.2f}", "short": True},
                        {"title": "Usage", "value": f"{alert.percentage_used:.1f}%", "short": True},
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def _send_email(self, alert: CostAlert) -> None:
        """Send alert via email"""
        # Implementation would depend on email service
        # This is a placeholder
        logger.info(f"Email alert would be sent: {alert.message}")
    
    async def _log_alert(self, alert: CostAlert) -> None:
        """Log alert locally"""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(self.log_file_path, 'a') as f:
                await f.write(json.dumps(alert.to_dict()) + "\n")
        except PermissionError as e:
            logger.error(f"Permission denied when logging alert to {self.log_file_path}: {e}")
        except OSError as e:
            logger.error(f"OS error when logging alert: {e}")
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def get_alert_history(self, since: Optional[datetime] = None) -> List[CostAlert]:
        """Get alert history, optionally filtered by time"""
        if since:
            return [a for a in self.alert_history if a.timestamp >= since]
        return self.alert_history.copy()