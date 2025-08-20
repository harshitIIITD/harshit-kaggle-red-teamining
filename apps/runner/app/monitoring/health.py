# ABOUTME: Comprehensive health monitoring and alerting system
# ABOUTME: Provides real-time health checks, performance monitoring, and automated alerting

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import json
import statistics
from pathlib import Path
import aiofiles
from abc import ABC, abstractmethod

from ..util.exceptions import (
    BaseRedTeamException, ResourceException, NetworkException,
    ErrorCode, ErrorSeverity, ErrorContext, error_tracker
)

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class MetricType(str, Enum):
    """Types of metrics to track"""
    COUNTER = "counter"      # Incrementing values
    GAUGE = "gauge"          # Current values
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"          # Time measurements


@dataclass
class HealthCheck:
    """Definition of a health check"""
    name: str
    check_function: Callable[[], Union[bool, Dict[str, Any]]]
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    warning_threshold: int = 2
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.consecutive_failures = 0
        self.last_check_time: Optional[datetime] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.status = HealthStatus.HEALTHY


@dataclass 
class Metric:
    """A single metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class Alert:
    """An alert notification"""
    id: str
    title: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "health_monitor"
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "tags": self.tags,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class AlertChannel(ABC):
    """Abstract base class for alert channels"""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through this channel"""
        pass


class LogAlertChannel(AlertChannel):
    """Log-based alert channel"""
    
    def __init__(self, logger_name: str = "alerts"):
        self.logger = logging.getLogger(logger_name)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to logger"""
        try:
            log_level = {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.INFO: logging.INFO
            }.get(alert.severity, logging.WARNING)
            
            self.logger.log(
                log_level,
                f"ALERT: {alert.title} - {alert.message}",
                extra={"alert_data": alert.to_dict()}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send log alert: {e}")
            return False


class FileAlertChannel(AlertChannel):
    """File-based alert channel"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to file"""
        try:
            alert_line = json.dumps(alert.to_dict()) + "\n"
            async with aiofiles.open(self.file_path, 'a') as f:
                await f.write(alert_line)
            return True
        except Exception as e:
            logger.error(f"Failed to send file alert: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel"""
    
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to webhook"""
        try:
            import httpx
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: List[Metric] = []
        self.max_metrics = max_metrics
        self._lock = asyncio.Lock()
    
    async def record_metric(self, metric: Metric):
        """Record a new metric"""
        async with self._lock:
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]
    
    async def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Metric]:
        """Get metrics matching criteria"""
        async with self._lock:
            filtered_metrics = self.metrics
            
            # Filter by name pattern
            if name_pattern:
                filtered_metrics = [
                    m for m in filtered_metrics 
                    if name_pattern in m.name
                ]
            
            # Filter by time
            if since:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.timestamp >= since
                ]
            
            # Apply limit
            if limit:
                filtered_metrics = filtered_metrics[-limit:]
            
            return filtered_metrics.copy()
    
    async def get_metric_summary(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        since = datetime.now(UTC) - timedelta(hours=hours)
        metrics = await self.get_metrics(name_pattern=name, since=since)
        
        if not metrics:
            return {"name": name, "count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1],
            "timestamp_range": {
                "start": metrics[0].timestamp.isoformat(),
                "end": metrics[-1].timestamp.isoformat()
            }
        }


class SystemMetricsCollector:
    """Collects system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.process = psutil.Process()
    
    async def collect_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            await self.metrics_collector.record_metric(Metric(
                name="system.cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.metrics_collector.record_metric(Metric(
                name="system.memory_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE
            ))
            
            await self.metrics_collector.record_metric(Metric(
                name="system.memory_available_mb",
                value=memory.available / 1024 / 1024,
                metric_type=MetricType.GAUGE
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.metrics_collector.record_metric(Metric(
                name="system.disk_percent",
                value=disk.percent,
                metric_type=MetricType.GAUGE
            ))
            
            # Process metrics
            process_memory = self.process.memory_info()
            await self.metrics_collector.record_metric(Metric(
                name="process.memory_rss_mb",
                value=process_memory.rss / 1024 / 1024,
                metric_type=MetricType.GAUGE
            ))
            
            await self.metrics_collector.record_metric(Metric(
                name="process.cpu_percent",
                value=self.process.cpu_percent(),
                metric_type=MetricType.GAUGE
            ))
            
            # File descriptor count
            try:
                fd_count = self.process.num_fds()
                await self.metrics_collector.record_metric(Metric(
                    name="process.file_descriptors",
                    value=fd_count,
                    metric_type=MetricType.GAUGE
                ))
            except (AttributeError, psutil.AccessDenied):
                # Not available on all platforms
                pass
            
            # Thread count
            thread_count = self.process.num_threads()
            await self.metrics_collector.record_metric(Metric(
                name="process.thread_count",
                value=thread_count,
                metric_type=MetricType.GAUGE
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_collector = MetricsCollector()
        self.system_metrics = SystemMetricsCollector(self.metrics_collector)
        self.alert_channels: List[AlertChannel] = []
        self.active_alerts: Dict[str, Alert] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        
        # Setup default alert channels
        self.add_alert_channel(LogAlertChannel())
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        # Memory usage check
        self.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval_seconds=30.0,
            failure_threshold=3,
            warning_threshold=2
        ))
        
        # Disk usage check
        self.add_health_check(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval_seconds=60.0,
            failure_threshold=2,
            warning_threshold=1
        ))
        
        # Database connectivity check
        self.add_health_check(HealthCheck(
            name="database_connectivity",
            check_function=self._check_database_connectivity,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            failure_threshold=3
        ))
        
        # Error rate check
        self.add_health_check(HealthCheck(
            name="error_rate",
            check_function=self._check_error_rate,
            interval_seconds=60.0,
            failure_threshold=2
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Removed health check: {name}")
    
    def add_alert_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        self.alert_channels.append(channel)
        logger.info(f"Added alert channel: {type(channel).__name__}")
    
    async def start(self):
        """Start the health monitoring system"""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting health monitoring system")
        
        # Start health check tasks
        for check in self.health_checks.values():
            if check.enabled:
                task = asyncio.create_task(self._run_health_check(check))
                self._tasks.append(task)
        
        # Start system metrics collection
        metrics_task = asyncio.create_task(self._collect_system_metrics())
        self._tasks.append(metrics_task)
        
        # Start alert cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_old_alerts())
        self._tasks.append(cleanup_task)
    
    async def stop(self):
        """Stop the health monitoring system"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping health monitoring system")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _run_health_check(self, check: HealthCheck):
        """Run a single health check continuously"""
        while self._running:
            try:
                # Wait for next interval
                await asyncio.sleep(check.interval_seconds)
                
                if not check.enabled:
                    continue
                
                # Run the check with timeout
                start_time = time.time()
                
                try:
                    result = await asyncio.wait_for(
                        self._execute_health_check(check),
                        timeout=check.timeout_seconds
                    )
                    duration = time.time() - start_time
                    
                    # Record timing metric
                    await self.metrics_collector.record_metric(Metric(
                        name=f"health_check.{check.name}.duration_ms",
                        value=duration * 1000,
                        metric_type=MetricType.TIMER,
                        tags={"check": check.name}
                    ))
                    
                    await self._process_health_check_result(check, result, True)
                    
                except asyncio.TimeoutError:
                    await self._process_health_check_result(
                        check, {"error": "Health check timed out"}, False
                    )
                
                except Exception as e:
                    await self._process_health_check_result(
                        check, {"error": str(e)}, False
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check {check.name}: {e}")
                await asyncio.sleep(min(check.interval_seconds, 60))  # Backoff
    
    async def _execute_health_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Execute a health check function"""
        if asyncio.iscoroutinefunction(check.check_function):
            result = await check.check_function()
        else:
            result = check.check_function()
        
        # Normalize result
        if isinstance(result, bool):
            return {"healthy": result}
        elif isinstance(result, dict):
            return result
        else:
            return {"healthy": True, "value": result}
    
    async def _process_health_check_result(
        self, 
        check: HealthCheck, 
        result: Dict[str, Any], 
        success: bool
    ):
        """Process health check result and update status"""
        check.last_check_time = datetime.now(UTC)
        check.last_result = result
        
        # Determine if check passed
        if success and result.get("healthy", True):
            # Check passed
            check.consecutive_failures = 0
            
            # Update status
            if check.status != HealthStatus.HEALTHY:
                old_status = check.status
                check.status = HealthStatus.HEALTHY
                
                # Resolve any active alerts
                await self._resolve_alert(f"health_check.{check.name}")
                
                logger.info(f"Health check {check.name} recovered: {old_status} -> {check.status}")
        else:
            # Check failed
            check.consecutive_failures += 1
            
            # Update status based on failure count
            old_status = check.status
            
            if check.consecutive_failures >= check.failure_threshold:
                check.status = HealthStatus.CRITICAL
            elif check.consecutive_failures >= check.warning_threshold:
                check.status = HealthStatus.WARNING
            else:
                check.status = HealthStatus.DEGRADED
            
            # Send alert if status changed or this is critical
            if (old_status != check.status or 
                check.status == HealthStatus.CRITICAL):
                
                await self._send_health_alert(check, result)
        
        # Record status metric
        status_value = {
            HealthStatus.HEALTHY: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.DEGRADED: 3,
            HealthStatus.CRITICAL: 4,
            HealthStatus.DOWN: 5
        }.get(check.status, 0)
        
        await self.metrics_collector.record_metric(Metric(
            name=f"health_check.{check.name}.status",
            value=status_value,
            metric_type=MetricType.GAUGE,
            tags={"check": check.name, "status": check.status.value}
        ))
    
    async def _send_health_alert(self, check: HealthCheck, result: Dict[str, Any]):
        """Send alert for health check failure"""
        alert_id = f"health_check.{check.name}"
        
        severity = {
            HealthStatus.WARNING: ErrorSeverity.MEDIUM,
            HealthStatus.DEGRADED: ErrorSeverity.HIGH,
            HealthStatus.CRITICAL: ErrorSeverity.CRITICAL,
            HealthStatus.DOWN: ErrorSeverity.CRITICAL
        }.get(check.status, ErrorSeverity.MEDIUM)
        
        alert = Alert(
            id=alert_id,
            title=f"Health Check Failed: {check.name}",
            message=f"Health check '{check.name}' has failed {check.consecutive_failures} "
                   f"consecutive times. Status: {check.status.value}. "
                   f"Details: {result}",
            severity=severity,
            tags={
                "check_name": check.name,
                "status": check.status.value,
                "consecutive_failures": str(check.consecutive_failures)
            }
        )
        
        async with self._lock:
            self.active_alerts[alert_id] = alert
        
        # Send through all channels
        for channel in self.alert_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through {type(channel).__name__}: {e}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        async with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = datetime.now(UTC)
                
                # Send resolution notification
                resolution_alert = Alert(
                    id=f"{alert_id}.resolved",
                    title=f"RESOLVED: {alert.title}",
                    message=f"Alert '{alert.title}' has been resolved.",
                    severity=ErrorSeverity.INFO,
                    tags=alert.tags
                )
                
                for channel in self.alert_channels:
                    try:
                        await channel.send_alert(resolution_alert)
                    except Exception as e:
                        logger.error(f"Failed to send resolution alert: {e}")
                
                del self.active_alerts[alert_id]
    
    async def _collect_system_metrics(self):
        """Continuously collect system metrics"""
        while self._running:
            try:
                await self.system_metrics.collect_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Backoff on error
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now(UTC) - timedelta(hours=24)
                
                # This would clean up persistent storage if we had it
                # For now, active alerts are cleaned up when resolved
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
    
    # Default health check implementations
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        memory = psutil.virtual_memory()
        
        return {
            "healthy": memory.percent < 90,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "warning_threshold": 80,
            "critical_threshold": 90
        }
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        disk = psutil.disk_usage('/')
        
        return {
            "healthy": disk.percent < 85,
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            "warning_threshold": 75,
            "critical_threshold": 85
        }
    
    async def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            from ..store.async_db import get_db_pool
            
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            return {"healthy": True, "message": "Database connection successful"}
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "message": "Database connection failed"
            }
    
    async def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate from error tracker"""
        try:
            stats = await error_tracker.get_error_stats()
            total_errors = stats.get("total_errors", 0)
            
            # Get errors from last hour
            one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
            recent_errors = len([
                error for error in error_tracker.errors
                if error.context and error.context.timestamp > one_hour_ago
            ])
            
            # Consider error rate critical if more than 50 errors in last hour
            error_rate_ok = recent_errors < 50
            
            return {
                "healthy": error_rate_ok,
                "recent_errors_1h": recent_errors,
                "total_errors": total_errors,
                "warning_threshold": 20,
                "critical_threshold": 50
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "message": "Failed to check error rate"
            }
    
    # Public API methods
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        overall_status = HealthStatus.HEALTHY
        check_results = {}
        
        for name, check in self.health_checks.items():
            check_results[name] = {
                "status": check.status.value,
                "consecutive_failures": check.consecutive_failures,
                "last_check": check.last_check_time.isoformat() if check.last_check_time else None,
                "last_result": check.last_result,
                "enabled": check.enabled
            }
            
            # Update overall status to worst individual status
            if check.status == HealthStatus.CRITICAL or check.status == HealthStatus.DOWN:
                overall_status = HealthStatus.CRITICAL
            elif check.status == HealthStatus.DEGRADED and overall_status not in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                overall_status = HealthStatus.DEGRADED
            elif check.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": check_results,
            "active_alerts": len(self.active_alerts),
            "metrics_count": len(self.metrics_collector.metrics)
        }
    
    async def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary"""
        summaries = {}
        
        # Common metrics to summarize
        metric_names = [
            "system.cpu_percent",
            "system.memory_percent", 
            "system.disk_percent",
            "process.memory_rss_mb",
            "process.cpu_percent"
        ]
        
        for metric_name in metric_names:
            summary = await self.metrics_collector.get_metric_summary(metric_name, hours)
            if summary.get("count", 0) > 0:
                summaries[metric_name] = summary
        
        return summaries
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        async with self._lock:
            return [alert.to_dict() for alert in self.active_alerts.values()]


# Global health monitor instance
health_monitor = HealthMonitor()


# Convenience functions for external use

async def add_custom_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, tags: Dict[str, str] = None):
    """Add a custom metric"""
    metric = Metric(
        name=name,
        value=value,
        metric_type=metric_type,
        tags=tags or {}
    )
    await health_monitor.metrics_collector.record_metric(metric)


async def send_custom_alert(title: str, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, tags: Dict[str, str] = None):
    """Send a custom alert"""
    alert = Alert(
        id=f"custom.{int(time.time() * 1000)}",
        title=title,
        message=message,
        severity=severity,
        tags=tags or {}
    )
    
    for channel in health_monitor.alert_channels:
        try:
            await channel.send_alert(alert)
        except Exception as e:
            logger.error(f"Failed to send custom alert: {e}")


# Decorator for timing function execution
def monitor_execution_time(metric_name: str, tags: Dict[str, str] = None):
    """Decorator to monitor function execution time"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                await add_custom_metric(
                    name=f"{metric_name}.duration_ms",
                    value=duration * 1000,
                    metric_type=MetricType.TIMER,
                    tags={**(tags or {}), "status": "success"}
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                await add_custom_metric(
                    name=f"{metric_name}.duration_ms",
                    value=duration * 1000,
                    metric_type=MetricType.TIMER,
                    tags={**(tags or {}), "status": "error"}
                )
                
                # Count errors
                await add_custom_metric(
                    name=f"{metric_name}.errors",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags={**(tags or {}), "error_type": type(e).__name__}
                )
                
                raise
        
        return wrapper
    return decorator