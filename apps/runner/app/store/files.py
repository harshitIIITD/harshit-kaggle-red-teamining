# ABOUTME: File storage utilities for managing JSONL transcripts and other file operations
# ABOUTME: Provides thread-safe append operations and rotation handling for JSONL files

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import fcntl
import re


# Global lock for file operations
_file_lock = threading.Lock()


def append_jsonl(filepath: Path, record: Dict[str, Any]) -> None:
    """
    Thread-safe append of a record to a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        record: Dictionary to append as JSON
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with _file_lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            # Use file locking for additional safety
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(record, f, ensure_ascii=False, default=str)
                f.write('\n')
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_jsonl_lines(filepath: Path) -> List[str]:
    """
    Read all lines from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of JSON strings (one per line)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def read_jsonl_records(filepath: Path) -> List[Dict[str, Any]]:
    """
    Read and parse all records from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of parsed dictionaries
    """
    lines = read_jsonl_lines(filepath)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines
            continue
    return records


def redact_pii(text: str) -> str:
    """
    Redact potential PII from text using regex patterns.
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Text with PII patterns replaced with [REDACTED] markers
    """
    if not text:
        return text
    
    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL]',
        text
    )
    
    # Phone numbers (various formats)
    text = re.sub(
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        '[PHONE]',
        text
    )
    # Additional phone pattern for xxx-xxxx-xxxx format
    text = re.sub(
        r'\b\d{3}-\d{4}-\d{4}\b',
        '[PHONE]',
        text
    )
    
    # SSN patterns
    text = re.sub(
        r'\b\d{3}-\d{2}-\d{4}\b',
        '[SSN]',
        text
    )
    
    # Credit card patterns (basic)
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '[CREDIT_CARD]',
        text
    )
    
    # IP addresses
    text = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '[IP_ADDRESS]',
        text
    )
    
    # AWS keys and similar
    text = re.sub(
        r'\b(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16}\b',
        '[AWS_KEY]',
        text
    )
    
    # Generic API key patterns
    text = re.sub(
        r'\b(?:api[_-]?key|apikey|api_secret|access[_-]?token)["\']?\s*[:=]\s*["\']?[A-Za-z0-9\-_]{20,}["\']?\b',
        '[API_KEY]',
        text,
        flags=re.IGNORECASE
    )
    
    return text


def rotate_file_if_needed(filepath: Path, max_size_mb: float = 100) -> Optional[Path]:
    """
    Rotate a file if it exceeds the maximum size.
    
    Args:
        filepath: Path to the file to check
        max_size_mb: Maximum size in megabytes before rotation
        
    Returns:
        Path to the rotated file if rotation occurred, None otherwise
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        # Create rotation name with timestamp
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rotated_path = filepath.parent / f"{filepath.stem}_{timestamp}{filepath.suffix}"
        
        with _file_lock:
            filepath.rename(rotated_path)
        
        return rotated_path
    
    return None


def ensure_directory(dirpath: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dirpath: Path to the directory
        
    Returns:
        The directory path
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath