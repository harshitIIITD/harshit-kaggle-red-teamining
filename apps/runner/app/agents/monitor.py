# ABOUTME: Performance monitoring and cost tracking for evaluation pipeline
# ABOUTME: Aggregates metrics from all components with alerting and budget management

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert with context"""
    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float


@dataclass
class CostBudget:
    """Cost budget configuration and tracking"""
    daily_limit_usd: float
    weekly_limit_usd: float
    monthly_limit_usd: float
    current_daily_spend: float = 0.0
    current_weekly_spend: float = 0.0
    current_monthly_spend: float = 0.0
    last_reset_daily: float = 0.0
    last_reset_weekly: float = 0.0
    last_reset_monthly: float = 0.0


@dataclass
class ComponentMetrics:
    """Metrics for a single component"""
    name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_time_ms: float = 0.0
    success_rate: float = 0.0
    recent_times: deque = None  # Last N timings for trend analysis
    
    def __post_init__(self):
        if self.recent_times is None:
            self.recent_times = deque(maxlen=100)  # Keep last 100 timings


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for evaluation pipeline.
    Tracks metrics, enforces budgets, and generates alerts.
    """
    
    # Performance targets for competition environment
    PERFORMANCE_TARGETS = {
        "avg_evaluation_time_ms": 2000,  # <2s average evaluation
        "heuristics_time_ms": 10,       # <10ms heuristics
        "cache_hit_rate_percent": 70,    # >70% cache hit rate
        "judge_escalation_rate_percent": 20,  # <20% escalation to judge
        "cost_reduction_percent": 80,    # 80% cost reduction target
        "success_rate_percent": 95       # >95% successful evaluations
    }
    
    # Alert thresholds (when to trigger warnings)
    ALERT_THRESHOLDS = {
        "avg_evaluation_time_ms": {"warning": 2500, "critical": 5000},
        "heuristics_time_ms": {"warning": 15, "critical": 25},
        "cache_hit_rate_percent": {"warning": 50, "critical": 30},
        "cost_per_hour_usd": {"warning": 5.0, "critical": 10.0},
        "success_rate_percent": {"warning": 90, "critical": 85}
    }
    
    def __init__(self, cost_budget: Optional[CostBudget] = None):
        """
        Initialize performance monitor.
        
        Args:
            cost_budget: Optional cost budget configuration
        """
        # Component metrics tracking
        self.components: Dict[str, ComponentMetrics] = {
            "heuristics": ComponentMetrics("heuristics"),
            "judge": ComponentMetrics("judge"), 
            "cache": ComponentMetrics("cache"),
            "evaluator": ComponentMetrics("evaluator")
        }
        
        # Cost budget management
        self.cost_budget = cost_budget or CostBudget(
            daily_limit_usd=50.0,    # Default budget limits
            weekly_limit_usd=300.0,
            monthly_limit_usd=1000.0
        )
        
        # Alert system
        self.alerts: List[PerformanceAlert] = []
        self.alert_cooldown: Dict[str, float] = {}  # Prevent spam
        self.alert_cooldown_seconds = 300  # 5 minute cooldown per alert type
        
        # System metrics
        self.start_time = time.time()
        self.total_evaluations = 0
        self.total_system_cost = 0.0
        
        logger.info("PerformanceMonitor initialized with budget tracking and alerting")
    
    def record_heuristics_call(self, time_ms: float, success: bool):
        """Record heuristics engine performance"""
        self._record_component_call("heuristics", time_ms, 0.0, success)
    
    def record_judge_call(self, time_ms: float, cost_usd: float, success: bool):
        """Record judge evaluation performance"""
        self._record_component_call("judge", time_ms, cost_usd, success)
    
    def record_cache_operation(self, time_ms: float, success: bool):
        """Record cache operation performance"""  
        self._record_component_call("cache", time_ms, 0.0, success)
    
    def record_evaluation(self, time_ms: float, cost_usd: float, success: bool):
        """Record overall evaluation performance"""
        self._record_component_call("evaluator", time_ms, cost_usd, success)
        self.total_evaluations += 1
        self.total_system_cost += cost_usd
    
    def _record_component_call(self, component_name: str, time_ms: float, 
                              cost_usd: float, success: bool):
        """Internal method to record component performance"""
        component = self.components[component_name]
        
        # Update counters
        component.total_calls += 1
        component.total_time_ms += time_ms
        component.total_cost_usd += cost_usd
        
        if success:
            component.success_count += 1
        else:
            component.failure_count += 1
        
        # Update derived metrics
        component.avg_time_ms = component.total_time_ms / component.total_calls
        component.success_rate = (component.success_count / component.total_calls) * 100
        
        # Track recent timings for trend analysis
        component.recent_times.append(time_ms)
        
        # Update budget tracking
        self._update_budget_tracking(cost_usd)
        
        # Check for performance issues
        self._check_performance_alerts(component_name, component)
    
    def _update_budget_tracking(self, cost_usd: float):
        """Update cost budget tracking with time-based resets"""
        current_time = time.time()
        
        # Reset daily budget if needed (24 hour cycle)
        if current_time - self.cost_budget.last_reset_daily >= 86400:  # 24 hours
            self.cost_budget.current_daily_spend = 0.0
            self.cost_budget.last_reset_daily = current_time
        
        # Reset weekly budget if needed (7 day cycle) 
        if current_time - self.cost_budget.last_reset_weekly >= 604800:  # 7 days
            self.cost_budget.current_weekly_spend = 0.0
            self.cost_budget.last_reset_weekly = current_time
        
        # Reset monthly budget if needed (30 day cycle)
        if current_time - self.cost_budget.last_reset_monthly >= 2592000:  # 30 days
            self.cost_budget.current_monthly_spend = 0.0
            self.cost_budget.last_reset_monthly = current_time
        
        # Add current cost to all budgets
        self.cost_budget.current_daily_spend += cost_usd
        self.cost_budget.current_weekly_spend += cost_usd
        self.cost_budget.current_monthly_spend += cost_usd
        
        # Check budget alerts
        self._check_budget_alerts()
    
    def _check_performance_alerts(self, component_name: str, component: ComponentMetrics):
        """Check component performance against thresholds and generate alerts"""
        current_time = time.time()
        
        # Check average time performance
        if component.avg_time_ms > 0:
            time_alert_key = f"{component_name}_avg_time"
            
            if component_name == "heuristics":
                target_key = "heuristics_time_ms"
            elif component_name == "evaluator":
                target_key = "avg_evaluation_time_ms"
            else:
                target_key = None
            
            if target_key and target_key in self.ALERT_THRESHOLDS:
                thresholds = self.ALERT_THRESHOLDS[target_key]
                
                if component.avg_time_ms >= thresholds["critical"]:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"{component_name} average time {component.avg_time_ms:.1f}ms exceeds critical threshold {thresholds['critical']}ms",
                        component_name, "avg_time_ms", component.avg_time_ms, thresholds["critical"], time_alert_key
                    )
                elif component.avg_time_ms >= thresholds["warning"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"{component_name} average time {component.avg_time_ms:.1f}ms exceeds warning threshold {thresholds['warning']}ms", 
                        component_name, "avg_time_ms", component.avg_time_ms, thresholds["warning"], time_alert_key
                    )
        
        # Check success rate
        if component.total_calls >= 10:  # Only after sufficient samples
            success_alert_key = f"{component_name}_success_rate"
            thresholds = self.ALERT_THRESHOLDS["success_rate_percent"]
            
            if component.success_rate <= thresholds["critical"]:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"{component_name} success rate {component.success_rate:.1f}% below critical threshold {thresholds['critical']}%",
                    component_name, "success_rate", component.success_rate, thresholds["critical"], success_alert_key
                )
            elif component.success_rate <= thresholds["warning"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"{component_name} success rate {component.success_rate:.1f}% below warning threshold {thresholds['warning']}%",
                    component_name, "success_rate", component.success_rate, thresholds["warning"], success_alert_key
                )
    
    def _check_budget_alerts(self):
        """Check cost budget limits and generate alerts"""
        current_time = time.time()
        
        # Daily budget check
        daily_usage_percent = (self.cost_budget.current_daily_spend / 
                              self.cost_budget.daily_limit_usd) * 100
        if daily_usage_percent >= 90:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Daily budget 90% consumed: ${self.cost_budget.current_daily_spend:.2f} / ${self.cost_budget.daily_limit_usd:.2f}",
                "budget", "daily_spend", self.cost_budget.current_daily_spend, 
                self.cost_budget.daily_limit_usd, "daily_budget"
            )
        elif daily_usage_percent >= 75:
            self._create_alert(
                AlertLevel.WARNING,
                f"Daily budget 75% consumed: ${self.cost_budget.current_daily_spend:.2f} / ${self.cost_budget.daily_limit_usd:.2f}",
                "budget", "daily_spend", self.cost_budget.current_daily_spend,
                self.cost_budget.daily_limit_usd, "daily_budget"
            )
        
        # Weekly budget check
        weekly_usage_percent = (self.cost_budget.current_weekly_spend /
                               self.cost_budget.weekly_limit_usd) * 100
        if weekly_usage_percent >= 90:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Weekly budget 90% consumed: ${self.cost_budget.current_weekly_spend:.2f} / ${self.cost_budget.weekly_limit_usd:.2f}",
                "budget", "weekly_spend", self.cost_budget.current_weekly_spend,
                self.cost_budget.weekly_limit_usd, "weekly_budget"
            )
    
    def _create_alert(self, level: AlertLevel, message: str, component: str,
                     metric_name: str, current_value: float, threshold: float, 
                     alert_key: str):
        """Create alert with cooldown to prevent spam"""
        current_time = time.time()
        
        # Check cooldown
        if alert_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[alert_key] < self.alert_cooldown_seconds:
                return  # Skip alert due to cooldown
        
        # Create and store alert
        alert = PerformanceAlert(
            level=level,
            message=message,
            component=component,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=current_time
        )
        
        self.alerts.append(alert)
        self.alert_cooldown[alert_key] = current_time
        
        # Log alert
        log_level = logging.CRITICAL if level == AlertLevel.CRITICAL else logging.WARNING
        logger.log(log_level, f"PERFORMANCE ALERT [{level.value.upper()}]: {message}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        current_time = time.time()
        uptime_hours = (current_time - self.start_time) / 3600
        
        # Component-level metrics
        component_metrics = {}
        for name, component in self.components.items():
            component_metrics[name] = {
                "total_calls": component.total_calls,
                "avg_time_ms": component.avg_time_ms,
                "total_cost_usd": component.total_cost_usd,
                "success_rate_percent": component.success_rate,
                "recent_trend_ms": self._calculate_trend(component.recent_times) if len(component.recent_times) >= 10 else None
            }
        
        # System-level metrics
        system_metrics = {
            "uptime_hours": uptime_hours,
            "total_evaluations": self.total_evaluations,
            "total_system_cost_usd": self.total_system_cost,
            "evaluations_per_hour": self.total_evaluations / max(uptime_hours, 0.1),
            "cost_per_evaluation_usd": self.total_system_cost / max(self.total_evaluations, 1),
            "cost_per_hour_usd": self.total_system_cost / max(uptime_hours, 0.1)
        }
        
        # Budget status
        budget_status = {
            "daily_budget_used_percent": (self.cost_budget.current_daily_spend / self.cost_budget.daily_limit_usd) * 100,
            "weekly_budget_used_percent": (self.cost_budget.current_weekly_spend / self.cost_budget.weekly_limit_usd) * 100,
            "monthly_budget_used_percent": (self.cost_budget.current_monthly_spend / self.cost_budget.monthly_limit_usd) * 100,
            "daily_remaining_usd": max(0, self.cost_budget.daily_limit_usd - self.cost_budget.current_daily_spend),
            "weekly_remaining_usd": max(0, self.cost_budget.weekly_limit_usd - self.cost_budget.current_weekly_spend),
            "monthly_remaining_usd": max(0, self.cost_budget.monthly_limit_usd - self.cost_budget.current_monthly_spend)
        }
        
        # Performance target compliance
        target_compliance = self._check_target_compliance(system_metrics, component_metrics)
        
        # Recent alerts (last 24 hours)
        cutoff_time = current_time - 86400  # 24 hours ago
        recent_alerts = [
            {
                "level": alert.level.value,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp,
                "age_hours": (current_time - alert.timestamp) / 3600
            }
            for alert in self.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        return {
            "timestamp": current_time,
            "system_metrics": system_metrics,
            "component_metrics": component_metrics,
            "budget_status": budget_status,
            "target_compliance": target_compliance,
            "recent_alerts": recent_alerts,
            "alert_summary": {
                "total_alerts": len(self.alerts),
                "recent_critical": len([a for a in recent_alerts if a["level"] == "critical"]),
                "recent_warnings": len([a for a in recent_alerts if a["level"] == "warning"])
            }
        }
    
    def _calculate_trend(self, recent_times: deque) -> str:
        """Calculate performance trend from recent timings"""
        if len(recent_times) < 10:
            return "insufficient_data"
        
        times_list = list(recent_times)
        first_half_avg = sum(times_list[:len(times_list)//2]) / (len(times_list)//2)
        second_half_avg = sum(times_list[len(times_list)//2:]) / (len(times_list) - len(times_list)//2)
        
        improvement_percent = ((first_half_avg - second_half_avg) / first_half_avg) * 100
        
        if improvement_percent > 10:
            return "improving"
        elif improvement_percent < -10:
            return "degrading"  
        else:
            return "stable"
    
    def _check_target_compliance(self, system_metrics: Dict, component_metrics: Dict) -> Dict[str, Any]:
        """Check compliance with performance targets"""
        compliance = {}
        
        # Check each target
        for target_name, target_value in self.PERFORMANCE_TARGETS.items():
            if target_name == "avg_evaluation_time_ms":
                current_value = component_metrics.get("evaluator", {}).get("avg_time_ms", 0)
                compliance[target_name] = {
                    "target": target_value,
                    "current": current_value,
                    "compliant": current_value <= target_value,
                    "deviation_percent": ((current_value - target_value) / target_value) * 100 if target_value > 0 else 0
                }
            elif target_name == "heuristics_time_ms":
                current_value = component_metrics.get("heuristics", {}).get("avg_time_ms", 0)
                compliance[target_name] = {
                    "target": target_value,
                    "current": current_value,
                    "compliant": current_value <= target_value,
                    "deviation_percent": ((current_value - target_value) / target_value) * 100 if target_value > 0 else 0
                }
        
        # Overall compliance score
        compliant_targets = sum(1 for t in compliance.values() if t["compliant"])
        compliance_score = (compliant_targets / len(compliance)) * 100 if compliance else 0
        
        compliance["overall"] = {
            "compliance_score_percent": compliance_score,
            "targets_met": compliant_targets,
            "total_targets": len(compliance)
        }
        
        return compliance
    
    def is_budget_exceeded(self, period: str = "daily") -> bool:
        """Check if budget is exceeded for given period"""
        if period == "daily":
            return self.cost_budget.current_daily_spend >= self.cost_budget.daily_limit_usd
        elif period == "weekly": 
            return self.cost_budget.current_weekly_spend >= self.cost_budget.weekly_limit_usd
        elif period == "monthly":
            return self.cost_budget.current_monthly_spend >= self.cost_budget.monthly_limit_usd
        return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alerts from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than N days"""
        cutoff_time = time.time() - (days * 86400)
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    async def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        metrics = self.get_comprehensive_metrics()
        
        report_lines = [
            "=== EVALUATION PIPELINE PERFORMANCE REPORT ===",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"System uptime: {metrics['system_metrics']['uptime_hours']:.1f} hours",
            "",
            "📊 SYSTEM METRICS:",
            f"  • Total evaluations: {metrics['system_metrics']['total_evaluations']:,}",
            f"  • Evaluations/hour: {metrics['system_metrics']['evaluations_per_hour']:.1f}",
            f"  • Cost per evaluation: ${metrics['system_metrics']['cost_per_evaluation_usd']:.4f}",
            f"  • Cost per hour: ${metrics['system_metrics']['cost_per_hour_usd']:.2f}",
            "",
            "⚡ COMPONENT PERFORMANCE:",
        ]
        
        for component_name, component_data in metrics["component_metrics"].items():
            if component_data["total_calls"] > 0:
                trend = component_data.get("recent_trend_ms", "unknown")
                report_lines.extend([
                    f"  📈 {component_name.upper()}:",
                    f"    - Avg time: {component_data['avg_time_ms']:.1f}ms (trend: {trend})",
                    f"    - Success rate: {component_data['success_rate_percent']:.1f}%",
                    f"    - Total cost: ${component_data['total_cost_usd']:.3f}"
                ])
        
        # Budget status
        report_lines.extend([
            "",
            "💰 BUDGET STATUS:",
            f"  • Daily: ${metrics['budget_status']['daily_remaining_usd']:.2f} remaining ({100-metrics['budget_status']['daily_budget_used_percent']:.1f}% available)",
            f"  • Weekly: ${metrics['budget_status']['weekly_remaining_usd']:.2f} remaining ({100-metrics['budget_status']['weekly_budget_used_percent']:.1f}% available)",
            f"  • Monthly: ${metrics['budget_status']['monthly_remaining_usd']:.2f} remaining ({100-metrics['budget_status']['monthly_budget_used_percent']:.1f}% available)"
        ])
        
        # Target compliance
        report_lines.extend([
            "",
            "🎯 TARGET COMPLIANCE:",
            f"  • Overall score: {metrics['target_compliance']['overall']['compliance_score_percent']:.1f}%",
            f"  • Targets met: {metrics['target_compliance']['overall']['targets_met']}/{metrics['target_compliance']['overall']['total_targets']}"
        ])
        
        # Recent alerts
        if metrics["recent_alerts"]:
            report_lines.extend([
                "",
                "🚨 RECENT ALERTS (24h):",
                f"  • Critical: {metrics['alert_summary']['recent_critical']}",
                f"  • Warnings: {metrics['alert_summary']['recent_warnings']}"
            ])
            
            for alert in metrics["recent_alerts"][-5:]:  # Show last 5 alerts
                report_lines.append(f"    - [{alert['level'].upper()}] {alert['message']} ({alert['age_hours']:.1f}h ago)")
        
        return "\n".join(report_lines)