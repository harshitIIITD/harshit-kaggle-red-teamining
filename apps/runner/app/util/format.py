# ABOUTME: Utility functions for formatting numbers, currency, and time values for UI display
# ABOUTME: Provides consistent formatting across dashboard and reports

from datetime import datetime, timezone
from typing import Union


def format_currency(amount: Union[int, float]) -> str:
    """Format a number as USD currency.
    
    Args:
        amount: Dollar amount to format
        
    Returns:
        Formatted string like "$1,234.56"
    """
    return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal value as percentage.
    
    Args:
        value: Decimal value (e.g., 0.123 for 12.3%)
        decimals: Number of decimal places
        
    Returns:
        Formatted string like "12.3%"
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """Format a number with thousands separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string like "1,234" or "1,234.56"
    """
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime as ISO timestamp.
    
    Args:
        dt: Datetime to format (defaults to current UTC time)
        
    Returns:
        ISO format string like "2024-01-20T15:30:45Z"
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 15m" or "45s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        remaining = seconds % 3600
        minutes = int(remaining / 60)
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"


def format_category_name(category: str) -> str:
    """Format category name for display.
    
    Args:
        category: Raw category name
        
    Returns:
        Human-readable category name
    """
    # Convert snake_case to Title Case
    return category.replace("_", " ").title()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."