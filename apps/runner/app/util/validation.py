# ABOUTME: Comprehensive input validation and sanitization framework
# ABOUTME: Provides security-focused validation, sanitization, and data integrity checks

import re
import html
import json
import base64
import hashlib
import logging
from typing import (
    Any, Dict, List, Optional, Union, Type, Callable, 
    Pattern, Set, Tuple, TypeVar, Generic
)
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import unicodedata
from pathlib import Path
import urllib.parse
import asyncio

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATION_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATION_AVAILABLE = False
    def validate_email(email): 
        raise NotImplementedError("email-validator not available")
    class EmailNotValidError(Exception):
        pass

try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False
    class bleach:
        @staticmethod
        def clean(text, tags=None, attributes=None, strip=False):
            return text  # Fallback: return unmodified text

from .exceptions import (
    ValidationException, SecurityException, BaseRedTeamException,
    ErrorCode, ErrorSeverity, ErrorContext, error_tracker
)

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation errors"""
    LOW = "low"           # Minor format issues
    MEDIUM = "medium"     # Data integrity issues  
    HIGH = "high"         # Security concerns
    CRITICAL = "critical" # Immediate security threats


class SanitizationMode(str, Enum):
    """Sanitization modes for different contexts"""
    STRICT = "strict"     # Remove all potentially dangerous content
    MODERATE = "moderate" # Allow safe HTML tags and attributes
    PERMISSIVE = "permissive" # Minimal sanitization
    CUSTOM = "custom"     # Use custom rules


@dataclass
class ValidationRule:
    """Definition of a validation rule"""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: ValidationSeverity = ValidationSeverity.MEDIUM
    sanitizer: Optional[Callable[[Any], Any]] = None
    required: bool = True
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value against this rule"""
        try:
            if self.validator(value):
                return True, None
            else:
                return False, self.error_message
        except Exception as e:
            return False, f"{self.error_message}: {str(e)}"
    
    def sanitize(self, value: Any) -> Any:
        """Sanitize a value using this rule's sanitizer"""
        if self.sanitizer:
            try:
                return self.sanitizer(value)
            except Exception as e:
                logger.warning(f"Sanitization failed for rule {self.name}: {e}")
        return value


@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Any = None
    original_data: Any = None
    severity: ValidationSeverity = ValidationSeverity.LOW
    
    def add_error(self, message: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM):
        """Add an error to the result"""
        self.errors.append(message)
        self.is_valid = False
        
        # Update overall severity to highest level
        severities = [ValidationSeverity.LOW, ValidationSeverity.MEDIUM, 
                     ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
        current_idx = severities.index(self.severity)
        new_idx = severities.index(severity)
        if new_idx > current_idx:
            self.severity = severity
    
    def add_warning(self, message: str):
        """Add a warning to the result"""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "severity": self.severity.value,
            "has_sanitized_data": self.sanitized_data is not None
        }


class SecurityPatterns:
    """Common security patterns for validation"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"('|(\\'))|(;|\s)+((union(\s)+(select))|" +
                  r"(select(.)*\*)|((update|delete|create|drop)(\s)+))", re.IGNORECASE),
        re.compile(r"(exec(\s)*\()|(\bexecute\b)", re.IGNORECASE),
        re.compile(r"(alter(\s)+)|((create|drop)(\s)+(table|database))", re.IGNORECASE),
        re.compile(r"(\bwhere\b.*\b(1=1|1\s*=\s*1)\b)", re.IGNORECASE)
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
        re.compile(r"<embed[^>]*>", re.IGNORECASE)
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\.[\\/]"),
        re.compile(r"[\\/]\.\."),
        re.compile(r"%2e%2e", re.IGNORECASE),
        re.compile(r"\.\.%2f", re.IGNORECASE),
        re.compile(r"%2f\.\."),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r"[;&|`$()]"),
        re.compile(r"\$\(.*\)"),
        re.compile(r"`.*`"),
        re.compile(r"&&|\|\|"),
    ]
    
    # Common dangerous strings
    DANGEROUS_STRINGS = {
        'drop table', 'delete from', 'truncate table', 'alter table',
        'create table', 'grant all', 'revoke all', 'exec xp_',
        'sp_executesql', 'xp_cmdshell', 'system(', 'exec(',
        '<script', 'javascript:', 'vbscript:', 'data:text/html',
        'eval(', 'expression(', 'url(javascript:', 'mocha:',
    }


class DataSanitizer:
    """Comprehensive data sanitization utilities"""
    
    @staticmethod
    def sanitize_string(
        value: str, 
        mode: SanitizationMode = SanitizationMode.MODERATE,
        max_length: Optional[int] = None,
        allowed_chars: Optional[Pattern] = None
    ) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            value = str(value)
        
        # Normalize unicode
        value = unicodedata.normalize('NFKC', value)
        
        # Remove null bytes and other dangerous characters
        value = value.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        
        # Apply length limit
        if max_length and len(value) > max_length:
            value = value[:max_length]
        
        # Apply character restrictions
        if allowed_chars:
            value = ''.join(c for c in value if allowed_chars.match(c))
        
        if mode == SanitizationMode.STRICT:
            # Remove all HTML and dangerous characters
            if BLEACH_AVAILABLE:
                value = bleach.clean(value, tags=[], attributes={}, strip=True)
            value = html.escape(value)
            
        elif mode == SanitizationMode.MODERATE:
            # Allow safe HTML tags
            if BLEACH_AVAILABLE:
                safe_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br', 'span']
                safe_attrs = {'span': ['style'], '*': ['class']}
                value = bleach.clean(value, tags=safe_tags, attributes=safe_attrs, strip=True)
            else:
                value = html.escape(value)
            
        elif mode == SanitizationMode.PERMISSIVE:
            # Minimal sanitization
            if BLEACH_AVAILABLE:
                value = bleach.clean(value, strip=False)
            else:
                # Fallback: escape HTML
                value = html.escape(value)
        
        return value.strip()
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """Sanitize HTML content"""
        if not BLEACH_AVAILABLE:
            return html.escape(value)
            
        safe_tags = [
            'p', 'br', 'strong', 'em', 'u', 'i', 'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'a', 'span', 'div'
        ]
        safe_attrs = {
            'a': ['href', 'title'],
            'span': ['style'],
            'div': ['class'],
            '*': ['class']
        }
        
        return bleach.clean(value, tags=safe_tags, attributes=safe_attrs, strip=True)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path separators and dangerous characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        filename = re.sub(r'\.\.+', '.', filename)  # Remove multiple dots
        filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
        
        # Ensure it's not empty
        if not filename:
            filename = 'unnamed_file'
        
        return filename[:255]  # Limit length
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL to prevent injection"""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Only allow safe schemes
            safe_schemes = {'http', 'https', 'ftp', 'ftps'}
            if parsed.scheme.lower() not in safe_schemes:
                raise ValidationException(
                    f"Unsafe URL scheme: {parsed.scheme}",
                    error_code=ErrorCode.VAL_INVALID_INPUT
                )
            
            # Reconstruct URL with safe components
            return urllib.parse.urlunparse(parsed)
            
        except Exception as e:
            raise ValidationException(
                f"Invalid URL format: {str(e)}",
                error_code=ErrorCode.VAL_FORMAT_ERROR
            )
    
    @staticmethod
    def sanitize_json(data: Any, max_depth: int = 10, max_items: int = 1000) -> Any:
        """Sanitize JSON data to prevent DoS attacks"""
        def _sanitize_recursive(obj, depth=0):
            if depth > max_depth:
                raise ValidationException(
                    f"JSON depth exceeds maximum: {depth} > {max_depth}",
                    error_code=ErrorCode.VAL_RANGE_ERROR
                )
            
            if isinstance(obj, dict):
                if len(obj) > max_items:
                    raise ValidationException(
                        f"JSON object has too many keys: {len(obj)} > {max_items}",
                        error_code=ErrorCode.VAL_RANGE_ERROR
                    )
                return {str(k): _sanitize_recursive(v, depth + 1) for k, v in obj.items()}
            
            elif isinstance(obj, list):
                if len(obj) > max_items:
                    raise ValidationException(
                        f"JSON array has too many items: {len(obj)} > {max_items}",
                        error_code=ErrorCode.VAL_RANGE_ERROR
                    )
                return [_sanitize_recursive(item, depth + 1) for item in obj]
            
            elif isinstance(obj, str):
                return DataSanitizer.sanitize_string(obj, SanitizationMode.MODERATE, 10000)
            
            else:
                return obj
        
        return _sanitize_recursive(data)


class SecurityValidator:
    """Security-focused validators"""
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns"""
        value_lower = value.lower()
        
        # Check dangerous strings
        for dangerous in SecurityPatterns.DANGEROUS_STRINGS:
            if dangerous in value_lower:
                return False
        
        # Check regex patterns
        for pattern in SecurityPatterns.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                return False
        
        return True
    
    @staticmethod
    def check_xss(value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in SecurityPatterns.XSS_PATTERNS:
            if pattern.search(value):
                return False
        return True
    
    @staticmethod
    def check_path_traversal(value: str) -> bool:
        """Check for path traversal patterns"""
        for pattern in SecurityPatterns.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                return False
        return True
    
    @staticmethod
    def check_command_injection(value: str) -> bool:
        """Check for command injection patterns"""
        for pattern in SecurityPatterns.COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                return False
        return True
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 10:
            return False
        
        # Check for basic format (alphanumeric with some special chars)
        if not re.match(r'^[a-zA-Z0-9._-]+$', api_key):
            return False
        
        return True
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """Validate model name format"""
        if not model_name or len(model_name) > 100:
            return False
        
        # Allow alphanumeric, hyphens, underscores, slashes, dots
        if not re.match(r'^[a-zA-Z0-9._/-]+$', model_name):
            return False
        
        return True


class DataValidator:
    """Comprehensive data validation framework"""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.global_rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        # Security rules
        self.add_global_rule(ValidationRule(
            name="sql_injection_check",
            validator=lambda x: SecurityValidator.check_sql_injection(str(x)) if x else True,
            error_message="Potential SQL injection detected",
            severity=ValidationSeverity.CRITICAL
        ))
        
        self.add_global_rule(ValidationRule(
            name="xss_check",
            validator=lambda x: SecurityValidator.check_xss(str(x)) if x else True,
            error_message="Potential XSS attack detected",
            severity=ValidationSeverity.HIGH
        ))
        
        self.add_global_rule(ValidationRule(
            name="path_traversal_check",
            validator=lambda x: SecurityValidator.check_path_traversal(str(x)) if x else True,
            error_message="Potential path traversal detected",
            severity=ValidationSeverity.HIGH
        ))
        
        # Field-specific rules
        self.add_rule("email", ValidationRule(
            name="email_format",
            validator=self._validate_email,
            error_message="Invalid email format",
            sanitizer=lambda x: x.lower().strip() if x else x
        ))
        
        self.add_rule("url", ValidationRule(
            name="url_format",
            validator=self._validate_url,
            error_message="Invalid URL format",
            sanitizer=DataSanitizer.sanitize_url
        ))
        
        self.add_rule("api_key", ValidationRule(
            name="api_key_format",
            validator=SecurityValidator.validate_api_key,
            error_message="Invalid API key format",
            severity=ValidationSeverity.HIGH
        ))
        
        self.add_rule("model_name", ValidationRule(
            name="model_name_format",
            validator=SecurityValidator.validate_model_name,
            error_message="Invalid model name format"
        ))
    
    def _validate_email(self, email: str) -> bool:
        """Validate email address"""
        if not EMAIL_VALIDATION_AVAILABLE:
            # Fallback regex validation
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
        
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """Add validation rule for specific field"""
        if field_name not in self.rules:
            self.rules[field_name] = []
        self.rules[field_name].append(rule)
    
    def add_global_rule(self, rule: ValidationRule):
        """Add global validation rule applied to all fields"""
        self.global_rules.append(rule)
    
    def validate_field(
        self, 
        field_name: str, 
        value: Any, 
        sanitize: bool = True
    ) -> ValidationResult:
        """Validate a single field"""
        result = ValidationResult(is_valid=True, original_data=value)
        
        if value is None:
            # Check if field is required
            field_rules = self.rules.get(field_name, [])
            if any(rule.required for rule in field_rules):
                result.add_error(f"Field '{field_name}' is required")
            return result
        
        sanitized_value = value
        
        # Apply global rules
        for rule in self.global_rules:
            is_valid, error_msg = rule.validate(value)
            if not is_valid:
                result.add_error(f"Global rule '{rule.name}': {error_msg}", rule.severity)
                
                # Track security violations
                if rule.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
                    security_error = SecurityException(
                        f"Security validation failed for field '{field_name}': {error_msg}",
                        error_code=ErrorCode.SEC_INJECTION_ATTEMPT,
                        severity=ErrorSeverity.HIGH,
                        context=ErrorContext(
                            operation="field_validation",
                            component="data_validator",
                            additional_data={
                                "field_name": field_name,
                                "rule_name": rule.name,
                                "value_preview": str(value)[:100] if value else None
                            }
                        )
                    )
                    asyncio.create_task(error_tracker.track_error(security_error))
            
            if sanitize and rule.sanitizer:
                sanitized_value = rule.sanitize(sanitized_value)
        
        # Apply field-specific rules
        field_rules = self.rules.get(field_name, [])
        for rule in field_rules:
            is_valid, error_msg = rule.validate(sanitized_value)
            if not is_valid:
                result.add_error(f"Field rule '{rule.name}': {error_msg}", rule.severity)
            
            if sanitize and rule.sanitizer:
                sanitized_value = rule.sanitize(sanitized_value)
        
        if sanitize:
            result.sanitized_data = sanitized_value
        
        return result
    
    def validate_dict(
        self, 
        data: Dict[str, Any], 
        sanitize: bool = True,
        strict: bool = False
    ) -> ValidationResult:
        """Validate dictionary data"""
        result = ValidationResult(is_valid=True, original_data=data)
        sanitized_data = {} if sanitize else None
        
        # Validate each field
        for field_name, value in data.items():
            field_result = self.validate_field(field_name, value, sanitize)
            
            if not field_result.is_valid:
                result.is_valid = False
                result.errors.extend([
                    f"{field_name}: {error}" for error in field_result.errors
                ])
                
                # Update severity
                if field_result.severity.value in ['high', 'critical']:
                    result.severity = field_result.severity
            
            result.warnings.extend([
                f"{field_name}: {warning}" for warning in field_result.warnings
            ])
            
            if sanitize:
                sanitized_data[field_name] = field_result.sanitized_data
        
        # Check for required fields that are missing
        if strict:
            for field_name, rules in self.rules.items():
                if field_name not in data:
                    required_rules = [rule for rule in rules if rule.required]
                    if required_rules:
                        result.add_error(f"Required field '{field_name}' is missing")
        
        if sanitize:
            result.sanitized_data = sanitized_data
        
        return result
    
    def validate_json_string(self, json_str: str, sanitize: bool = True) -> ValidationResult:
        """Validate and parse JSON string"""
        result = ValidationResult(is_valid=True, original_data=json_str)
        
        try:
            # Parse JSON
            data = json.loads(json_str)
            
            # Sanitize if requested
            if sanitize:
                data = DataSanitizer.sanitize_json(data)
            
            # Validate structure if it's a dict
            if isinstance(data, dict):
                dict_result = self.validate_dict(data, sanitize=False)  # Already sanitized
                result.is_valid = dict_result.is_valid
                result.errors = dict_result.errors
                result.warnings = dict_result.warnings
                result.severity = dict_result.severity
            
            result.sanitized_data = data
            
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {str(e)}")
        except ValidationException as e:
            result.add_error(f"JSON validation failed: {str(e)}", ValidationSeverity.HIGH)
        
        return result


# Global validator instance
global_validator = DataValidator()


# Convenience functions for common validation tasks

def validate_api_request(data: Dict[str, Any], sanitize: bool = True) -> ValidationResult:
    """Validate API request data"""
    return global_validator.validate_dict(data, sanitize, strict=True)


def validate_user_input(text: str, max_length: int = 1000) -> ValidationResult:
    """Validate user text input"""
    result = ValidationResult(is_valid=True, original_data=text)
    
    if not text:
        result.add_error("Input cannot be empty")
        return result
    
    if len(text) > max_length:
        result.add_error(f"Input too long: {len(text)} > {max_length} characters")
    
    # Security checks
    if not SecurityValidator.check_sql_injection(text):
        result.add_error("Potential SQL injection detected", ValidationSeverity.CRITICAL)
    
    if not SecurityValidator.check_xss(text):
        result.add_error("Potential XSS attack detected", ValidationSeverity.HIGH)
    
    if not SecurityValidator.check_command_injection(text):
        result.add_error("Potential command injection detected", ValidationSeverity.HIGH)
    
    # Sanitize if valid
    if result.is_valid:
        result.sanitized_data = DataSanitizer.sanitize_string(
            text, SanitizationMode.MODERATE, max_length
        )
    
    return result


def validate_file_upload(
    filename: str, 
    content: bytes, 
    allowed_extensions: Set[str] = None,
    max_size: int = 10 * 1024 * 1024  # 10MB default
) -> ValidationResult:
    """Validate file upload"""
    result = ValidationResult(is_valid=True, original_data={"filename": filename, "size": len(content)})
    
    # Validate filename
    if not filename:
        result.add_error("Filename cannot be empty")
        return result
    
    # Check for path traversal
    if not SecurityValidator.check_path_traversal(filename):
        result.add_error("Unsafe filename detected", ValidationSeverity.HIGH)
    
    # Sanitize filename
    safe_filename = DataSanitizer.sanitize_filename(filename)
    
    # Check extension
    if allowed_extensions:
        ext = Path(safe_filename).suffix.lower()
        if ext not in allowed_extensions:
            result.add_error(f"File extension {ext} not allowed")
    
    # Check file size
    if len(content) > max_size:
        result.add_error(f"File too large: {len(content)} > {max_size} bytes")
    
    # Check for executable content (basic)
    if content.startswith(b'\x7fELF') or content.startswith(b'MZ'):
        result.add_error("Executable files are not allowed", ValidationSeverity.HIGH)
    
    if result.is_valid:
        result.sanitized_data = {"filename": safe_filename, "content": content}
    
    return result


# Decorator for automatic input validation
def validate_input(
    sanitize: bool = True,
    strict: bool = True,
    custom_validator: Optional[DataValidator] = None
):
    """Decorator for automatic input validation"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            validator = custom_validator or global_validator
            
            # Validate kwargs (assuming they contain the input data)
            if kwargs:
                result = validator.validate_dict(kwargs, sanitize, strict)
                
                if not result.is_valid:
                    raise ValidationException(
                        f"Input validation failed: {'; '.join(result.errors)}",
                        error_code=ErrorCode.VAL_INVALID_INPUT,
                        severity=ErrorSeverity.HIGH if result.severity.value in ['high', 'critical'] else ErrorSeverity.MEDIUM,
                        context=ErrorContext(
                            operation="input_validation",
                            component="validation_decorator",
                            additional_data={
                                "function": func.__name__,
                                "validation_errors": result.errors,
                                "validation_warnings": result.warnings
                            }
                        )
                    )
                
                # Use sanitized data if available
                if sanitize and result.sanitized_data:
                    kwargs.update(result.sanitized_data)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator