# ABOUTME: Comprehensive exception handling framework for robust error management
# ABOUTME: Defines custom exceptions, error codes, and standardized error handling patterns

import logging
import traceback
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, UTC
import asyncio
import json


class ErrorSeverity(str, Enum):
    """Error severity levels for classification and handling"""
    CRITICAL = "critical"    # System-wide failures, immediate attention required
    HIGH = "high"           # Feature failures, affects core functionality
    MEDIUM = "medium"       # Degraded performance, partial functionality
    LOW = "low"            # Minor issues, logging only
    INFO = "info"          # Informational, not errors


class ErrorCategory(str, Enum):
    """Error categories for better classification and handling"""
    NETWORK = "network"              # Network connectivity, API failures
    DATABASE = "database"            # Database connection, query failures
    FILESYSTEM = "filesystem"        # File I/O, permission errors
    AUTHENTICATION = "authentication" # API key, credential issues
    VALIDATION = "validation"        # Input validation, data format errors
    RESOURCE = "resource"            # Memory, disk space, rate limits
    CONFIGURATION = "configuration"  # Config file, environment issues
    EXTERNAL_API = "external_api"    # Third-party API failures
    BUSINESS_LOGIC = "business_logic" # Application logic errors
    SECURITY = "security"            # Security-related issues
    UNKNOWN = "unknown"              # Unclassified errors


class ErrorCode(str, Enum):
    """Standardized error codes for precise error identification"""
    # Network errors
    NETWORK_TIMEOUT = "NET_001"
    NETWORK_CONNECTION_FAILED = "NET_002"
    NETWORK_DNS_RESOLUTION = "NET_003"
    NETWORK_SSL_ERROR = "NET_004"
    
    # Database errors
    DB_CONNECTION_FAILED = "DB_001"
    DB_QUERY_FAILED = "DB_002"
    DB_TIMEOUT = "DB_003"
    DB_LOCK_TIMEOUT = "DB_004"
    DB_CORRUPTION = "DB_005"
    
    # Filesystem errors
    FS_PERMISSION_DENIED = "FS_001"
    FS_FILE_NOT_FOUND = "FS_002"
    FS_DISK_FULL = "FS_003"
    FS_CORRUPTION = "FS_004"
    FS_ACCESS_DENIED = "FS_005"
    
    # Authentication errors
    AUTH_INVALID_KEY = "AUTH_001"
    AUTH_EXPIRED_TOKEN = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"
    AUTH_RATE_LIMITED = "AUTH_004"
    
    # Validation errors
    VAL_INVALID_INPUT = "VAL_001"
    VAL_MISSING_REQUIRED = "VAL_002"
    VAL_FORMAT_ERROR = "VAL_003"
    VAL_RANGE_ERROR = "VAL_004"
    
    # Resource errors
    RES_MEMORY_EXHAUSTED = "RES_001"
    RES_DISK_FULL = "RES_002"
    RES_RATE_LIMIT_EXCEEDED = "RES_003"
    RES_QUOTA_EXCEEDED = "RES_004"
    
    # Configuration errors
    CFG_MISSING_FILE = "CFG_001"
    CFG_INVALID_FORMAT = "CFG_002"
    CFG_MISSING_KEY = "CFG_003"
    CFG_INVALID_VALUE = "CFG_004"
    
    # External API errors
    API_UNAVAILABLE = "API_001"
    API_RATE_LIMITED = "API_002"
    API_INVALID_RESPONSE = "API_003"
    API_TIMEOUT = "API_004"
    
    # Business logic errors
    BL_INVALID_STATE = "BL_001"
    BL_OPERATION_FAILED = "BL_002"
    BL_CONSTRAINT_VIOLATION = "BL_003"
    BL_DATA_INCONSISTENCY = "BL_004"
    
    # Security errors
    SEC_UNAUTHORIZED_ACCESS = "SEC_001"
    SEC_INJECTION_ATTEMPT = "SEC_002"
    SEC_SUSPICIOUS_ACTIVITY = "SEC_003"
    SEC_DATA_BREACH = "SEC_004"
    
    # Generic errors
    UNKNOWN_ERROR = "GEN_001"
    INTERNAL_ERROR = "GEN_002"
    TIMEOUT = "GEN_003"
    CANCELLED = "GEN_004"


@dataclass
class ErrorContext:
    """Rich context information for error tracking and debugging"""
    error_id: str = field(default_factory=lambda: f"err_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    operation: Optional[str] = None
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "additional_data": self.additional_data,
            "stack_trace": self.stack_trace
        }


class BaseRedTeamException(Exception):
    """Base exception class for all red-teaming system errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        retry_after: Optional[float] = None,
        should_retry: bool = False,
        max_retries: int = 3,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.retry_after = retry_after
        self.should_retry = should_retry
        self.max_retries = max_retries
        self.original_exception = original_exception
        
        # Capture stack trace
        if self.context and not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses"""
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict() if self.context else None,
            "retry_after": self.retry_after,
            "should_retry": self.should_retry,
            "max_retries": self.max_retries,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


# Specific exception classes for different categories

class NetworkException(BaseRedTeamException):
    """Network-related exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.NETWORK_CONNECTION_FAILED, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.NETWORK, 
            should_retry=True,
            **kwargs
        )


class DatabaseException(BaseRedTeamException):
    """Database-related exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DB_CONNECTION_FAILED, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.DATABASE,
            should_retry=True,
            **kwargs
        )


class FilesystemException(BaseRedTeamException):
    """Filesystem-related exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.FS_PERMISSION_DENIED, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.FILESYSTEM,
            should_retry=False,
            **kwargs
        )


class AuthenticationException(BaseRedTeamException):
    """Authentication and authorization exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.AUTH_INVALID_KEY, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            should_retry=False,
            **kwargs
        )


class ValidationException(BaseRedTeamException):
    """Input validation and data format exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.VAL_INVALID_INPUT, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            should_retry=False,
            **kwargs
        )


class ResourceException(BaseRedTeamException):
    """Resource exhaustion and limits exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.RES_MEMORY_EXHAUSTED, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            should_retry=True,
            **kwargs
        )


class ConfigurationException(BaseRedTeamException):
    """Configuration and environment exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.CFG_MISSING_FILE, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            should_retry=False,
            **kwargs
        )


class ExternalAPIException(BaseRedTeamException):
    """External API and service exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.API_UNAVAILABLE, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.EXTERNAL_API,
            should_retry=True,
            **kwargs
        )


class BusinessLogicException(BaseRedTeamException):
    """Business logic and application flow exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.BL_INVALID_STATE, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.BUSINESS_LOGIC,
            should_retry=False,
            **kwargs
        )


class SecurityException(BaseRedTeamException):
    """Security-related exceptions"""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.SEC_UNAUTHORIZED_ACCESS, **kwargs):
        super().__init__(
            message, 
            error_code=error_code, 
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            should_retry=False,
            **kwargs
        )


# Error tracking and metrics
class ErrorTracker:
    """Centralized error tracking and metrics collection"""
    
    def __init__(self):
        self.errors: List[BaseRedTeamException] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def track_error(self, error: BaseRedTeamException):
        """Track an error occurrence"""
        async with self._lock:
            self.errors.append(error)
            
            # Update counts
            error_key = f"{error.category.value}:{error.error_code.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Log the error
            log_level = {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.INFO: logging.INFO
            }.get(error.severity, logging.WARNING)
            
            self.logger.log(
                log_level,
                f"Error tracked: {error}",
                extra={"error_data": error.to_dict()}
            )
    
    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        async with self._lock:
            total_errors = len(self.errors)
            
            severity_counts = {}
            category_counts = {}
            
            for error in self.errors:
                # Count by severity
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                
                # Count by category
                category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            
            return {
                "total_errors": total_errors,
                "severity_breakdown": severity_counts,
                "category_breakdown": category_counts,
                "error_counts": self.error_counts.copy(),
                "recent_errors": [error.to_dict() for error in self.errors[-10:]]  # Last 10 errors
            }
    
    async def clear_old_errors(self, max_age_hours: int = 24):
        """Clear errors older than specified hours"""
        async with self._lock:
            cutoff_time = datetime.now(UTC).timestamp() - (max_age_hours * 3600)
            
            self.errors = [
                error for error in self.errors
                if error.context and error.context.timestamp.timestamp() > cutoff_time
            ]


# Global error tracker instance
error_tracker = ErrorTracker()


# Utility functions for error handling

def exception_to_error_code(exception: Exception) -> ErrorCode:
    """Map common exceptions to error codes"""
    mapping = {
        ConnectionError: ErrorCode.NETWORK_CONNECTION_FAILED,
        TimeoutError: ErrorCode.NETWORK_TIMEOUT,
        PermissionError: ErrorCode.FS_PERMISSION_DENIED,
        FileNotFoundError: ErrorCode.FS_FILE_NOT_FOUND,
        OSError: ErrorCode.FS_ACCESS_DENIED,
        ValueError: ErrorCode.VAL_INVALID_INPUT,
        KeyError: ErrorCode.VAL_MISSING_REQUIRED,
        asyncio.TimeoutError: ErrorCode.TIMEOUT,
        asyncio.CancelledError: ErrorCode.CANCELLED,
    }
    
    for exc_type, error_code in mapping.items():
        if isinstance(exception, exc_type):
            return error_code
    
    return ErrorCode.UNKNOWN_ERROR


def create_error_from_exception(
    exception: Exception,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> BaseRedTeamException:
    """Create a standardized error from a generic exception"""
    
    error_code = exception_to_error_code(exception)
    
    context = ErrorContext(
        operation=operation,
        component=component,
        additional_data=additional_context or {}
    )
    
    # Map to appropriate exception type
    if error_code in [ErrorCode.NETWORK_CONNECTION_FAILED, ErrorCode.NETWORK_TIMEOUT]:
        return NetworkException(
            str(exception),
            error_code=error_code,
            context=context,
            original_exception=exception
        )
    elif error_code in [ErrorCode.FS_PERMISSION_DENIED, ErrorCode.FS_FILE_NOT_FOUND]:
        return FilesystemException(
            str(exception),
            error_code=error_code,
            context=context,
            original_exception=exception
        )
    elif error_code in [ErrorCode.VAL_INVALID_INPUT, ErrorCode.VAL_MISSING_REQUIRED]:
        return ValidationException(
            str(exception),
            error_code=error_code,
            context=context,
            original_exception=exception
        )
    else:
        return BaseRedTeamException(
            str(exception),
            error_code=error_code,
            context=context,
            original_exception=exception
        )


def log_and_track_error(
    error: BaseRedTeamException,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log and track an error (synchronous wrapper)"""
    if logger:
        logger.error(f"Error occurred: {error}", extra={"error_data": error.to_dict()})
    
    # Schedule async tracking
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(error_tracker.track_error(error))
    except RuntimeError:
        # No event loop running, just log
        logging.getLogger(__name__).error(f"Failed to track error: {error}")


# Decorators for error handling

def handle_exceptions(
    component: str,
    operation: Optional[str] = None,
    reraise: bool = True,
    default_return=None
):
    """Decorator for standardized exception handling"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except BaseRedTeamException:
                    # Already handled, just reraise
                    raise
                except Exception as e:
                    error = create_error_from_exception(
                        e,
                        operation=operation or func.__name__,
                        component=component
                    )
                    await error_tracker.track_error(error)
                    
                    if reraise:
                        raise error
                    return default_return
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except BaseRedTeamException:
                    # Already handled, just reraise
                    raise
                except Exception as e:
                    error = create_error_from_exception(
                        e,
                        operation=operation or func.__name__,
                        component=component
                    )
                    log_and_track_error(error)
                    
                    if reraise:
                        raise error
                    return default_return
            return sync_wrapper
    return decorator