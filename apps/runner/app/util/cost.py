# ABOUTME: Cost calculation utilities for tracking API usage expenses
# ABOUTME: Maps token usage to dollar amounts based on model pricing

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Default cost cap in USD
DEFAULT_COST_CAP = 10.0
SMOKE_TEST_COST_CAP = 1.0


class CostTracker:
    """Track cumulative costs and enforce caps"""

    def __init__(self, cost_cap: float = DEFAULT_COST_CAP):
        self.cost_cap = cost_cap
        self.total_cost = 0.0
        self.attempts = 0
        self.total_tokens = 0

    def add_attempt(self, usage: Dict[str, Any]) -> bool:
        """Add attempt cost and check if under cap

        Returns:
            bool: True if still under cap, False if cap exceeded
        """
        cost = usage.get("cost_usd", 0.0)
        self.total_cost += cost
        self.total_tokens += usage.get("total_tokens", 0)
        self.attempts += 1

        if self.total_cost > self.cost_cap:
            logger.warning(
                f"Cost cap exceeded: ${self.total_cost:.2f} > ${self.cost_cap:.2f}"
            )
            return False

        return True

    def get_remaining_budget(self) -> float:
        """Get remaining budget under cap"""
        return max(0, self.cost_cap - self.total_cost)

    def get_average_cost(self) -> float:
        """Get average cost per attempt"""
        if self.attempts == 0:
            return 0.0
        return self.total_cost / self.attempts

    def get_stats(self) -> Dict[str, Any]:
        """Get cost tracking statistics"""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "cost_cap_usd": self.cost_cap,
            "remaining_budget_usd": round(self.get_remaining_budget(), 4),
            "total_attempts": self.attempts,
            "average_cost_usd": round(self.get_average_cost(), 4),
            "total_tokens": self.total_tokens,
            "under_cap": self.total_cost <= self.cost_cap,
        }

    def reset(self):
        """Reset all counters"""
        self.total_cost = 0.0
        self.attempts = 0
        self.total_tokens = 0


def estimate_prompt_cost(
    prompt: str, model: str = "llama3", response_tokens: int = 500
) -> float:
    """Estimate cost for a prompt before sending (returns 0.0 for local Ollama inference)

    Args:
        prompt: The prompt text
        model: Model identifier
        response_tokens: Expected response tokens

    Returns:
        Estimated cost in USD (always 0.0 for local inference)
    """
    # No cost for local inference
    return 0.0


def estimate_cost(model: str, usage: Dict[str, Any]) -> float:
    """Calculate actual cost from usage information (returns 0.0 for local Ollama inference)
    
    Args:
        model: Model identifier
        usage: Usage dictionary with token counts
        
    Returns:
        Cost in USD (always 0.0 for local inference)
    """
    # No cost for local inference
    return 0.0
