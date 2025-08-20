# ABOUTME: Input sanitization to prevent adversarial manipulation of judge models
# ABOUTME: Removes dangerous patterns that could influence judge evaluation results

import re


class InputSanitizer:
    """
    Sanitizes inputs before sending to judge models to prevent adversarial manipulation.
    Removes or redacts patterns that could influence the judge's evaluation.
    """
    
    # Patterns that could manipulate judge behavior
    DANGEROUS_PATTERNS = [
        # System prompt injection attempts
        (r'</?system[^>]*>', '[REDACTED_SYSTEM_TAG]'),
        (r'</?assistant[^>]*>', '[REDACTED_ASSISTANT_TAG]'),
        (r'</?user[^>]*>', '[REDACTED_USER_TAG]'),
        
        # Direct instruction override attempts  
        (r'\b(?:ignore|disregard|forget)\s+(?:previous|all|above)\s+(?:instructions|context|rules|guidelines)\b', '[REDACTED_OVERRIDE]'),
        (r'\b(?:you\s+are\s+now|from\s+now\s+on|instead\s+of)\s+(?:a|an|the)', '[REDACTED_ROLE_CHANGE]'),
        
        # Judge manipulation attempts
        (r'\b(?:rate\s+this|judge\s+this|evaluate\s+this)\s+(?:as|to\s+be)\s+(?:safe|harmless|benign)', '[REDACTED_EVAL_BIAS]'),
        (r'\b(?:this\s+is|consider\s+this)\s+(?:safe|harmless|not\s+dangerous)', '[REDACTED_SAFETY_CLAIM]'),
        
        # Meta-commentary that could bias evaluation
        (r'\b(?:for\s+(?:educational|research|academic)\s+purposes?\s+only)\b', '[REDACTED_PURPOSE_CLAIM]'),
        (r'\b(?:this\s+is\s+(?:just\s+a\s+)?(?:hypothetical|theoretical|fictional))\b', '[REDACTED_FICTIONAL_CLAIM]'),
        
        # Judge context poisoning
        (r'\b(?:the\s+(?:above|previous)\s+(?:content|text|response)\s+is\s+(?:safe|harmless))\b', '[REDACTED_CONTEXT_POISON]'),
    ]
    
    # Compiled patterns for performance
    _compiled_patterns = None
    
    def __init__(self, max_length: int = 7000):
        """
        Initialize sanitizer with maximum text length.
        Args:
            max_length: Maximum length to prevent token limit issues
        """
        self.max_length = max_length
        
        # Compile patterns once for performance
        if InputSanitizer._compiled_patterns is None:
            InputSanitizer._compiled_patterns = [
                (re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement)
                for pattern, replacement in self.DANGEROUS_PATTERNS
            ]
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize input text to prevent adversarial manipulation.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text safe for judge evaluation
        """
        if not text or not text.strip():
            return text
        
        sanitized = text
        
        # Apply all dangerous pattern replacements
        for compiled_pattern, replacement in self._compiled_patterns:
            sanitized = compiled_pattern.sub(replacement, sanitized)
        
        # Truncate to prevent token limit issues
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length] + "[TRUNCATED]"
        
        return sanitized
    
    def is_potentially_adversarial(self, text: str) -> bool:
        """
        Check if text contains potentially adversarial patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if potentially adversarial patterns detected
        """
        if not text:
            return False
        
        for compiled_pattern, _ in self._compiled_patterns:
            if compiled_pattern.search(text):
                return True
        
        return False
    
    def get_sanitization_report(self, original: str, sanitized: str) -> dict:
        """
        Generate report of sanitization actions taken.
        
        Args:
            original: Original text
            sanitized: Sanitized text
            
        Returns:
            Dictionary with sanitization details
        """
        changes = []
        temp_text = original
        
        for compiled_pattern, replacement in self._compiled_patterns:
            matches = compiled_pattern.findall(temp_text)
            if matches:
                changes.append({
                    "pattern_type": replacement.strip("[]"),
                    "match_count": len(matches),
                    "matches": matches[:3]  # First 3 matches for debugging
                })
                temp_text = compiled_pattern.sub(replacement, temp_text)
        
        return {
            "original_length": len(original),
            "sanitized_length": len(sanitized),
            "was_truncated": len(sanitized) < len(temp_text),
            "changes_made": len(changes),
            "sanitization_details": changes
        }