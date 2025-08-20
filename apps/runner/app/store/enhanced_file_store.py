# ABOUTME: Enhanced file storage operations with comprehensive error handling and recovery
# ABOUTME: Provides robust file I/O with corruption detection, atomic operations, and monitoring

import os
import json
import asyncio
import aiofiles
import time
import shutil
import hashlib
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from pathlib import Path
from datetime import datetime, UTC
import logging
import tempfile
import fcntl
from contextlib import asynccontextmanager

from ..util.exceptions import (
    BaseRedTeamException, FilesystemException, ValidationException,
    ErrorCode, ErrorSeverity, ErrorContext, error_tracker, handle_exceptions
)
from ..util.retry import database_retry_handler, RetryConfig
from ..util.validation import validate_file_upload, global_validator
from ..monitoring.health import add_custom_metric, MetricType

logger = logging.getLogger(__name__)


class FileCorruptionError(FilesystemException):
    """Raised when file corruption is detected"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.FS_CORRUPTION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class FileLockManager:
    """Manages file locking for safe concurrent access"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lock_path = self.file_path.with_suffix(f"{self.file_path.suffix}.lock")
        self._lock_fd: Optional[int] = None
    
    @asynccontextmanager
    async def acquire_lock(self, timeout: float = 30.0):
        """Acquire exclusive file lock"""
        start_time = time.time()
        
        try:
            # Create lock file if it doesn't exist
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open lock file
            self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_WRONLY)
            
            # Try to acquire lock with timeout
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.debug(f"Acquired file lock: {self.lock_path}")
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.1)
            else:
                raise FilesystemException(
                    f"Failed to acquire file lock within {timeout}s",
                    error_code=ErrorCode.FS_ACCESS_DENIED,
                    context=ErrorContext(
                        operation="file_lock_acquisition",
                        component="file_lock_manager",
                        additional_data={"file_path": str(self.file_path)}
                    )
                )
            
            yield
            
        finally:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    os.close(self._lock_fd)
                    logger.debug(f"Released file lock: {self.lock_path}")
                except Exception as e:
                    logger.warning(f"Error releasing file lock: {e}")
                finally:
                    self._lock_fd = None
                    
                    # Clean up lock file
                    try:
                        self.lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass


class FileIntegrityChecker:
    """Checks file integrity using checksums and validation"""
    
    @staticmethod
    def calculate_checksum(data: bytes) -> str:
        """Calculate SHA-256 checksum of data"""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def calculate_file_checksum(file_path: Path) -> str:
        """Calculate checksum of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def validate_jsonl_structure(content: str) -> List[str]:
        """Validate JSONL file structure and return error messages"""
        errors = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
        
        return errors
    
    @classmethod
    async def validate_file_integrity(
        cls, 
        file_path: Path, 
        expected_checksum: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive file integrity validation"""
        try:
            if not file_path.exists():
                return {
                    "valid": False,
                    "error": "File does not exist",
                    "file_exists": False
                }
            
            # Calculate current checksum
            current_checksum = cls.calculate_file_checksum(file_path)
            
            # Check against expected checksum if provided
            checksum_valid = True
            if expected_checksum and current_checksum != expected_checksum:
                checksum_valid = False
            
            # Get file stats
            stat = file_path.stat()
            
            # Additional validation for JSONL files
            structure_errors = []
            if file_path.suffix.lower() == '.jsonl':
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                    structure_errors = cls.validate_jsonl_structure(content)
                except Exception as e:
                    structure_errors = [f"Failed to read file: {str(e)}"]
            
            return {
                "valid": checksum_valid and len(structure_errors) == 0,
                "file_exists": True,
                "current_checksum": current_checksum,
                "expected_checksum": expected_checksum,
                "checksum_valid": checksum_valid,
                "structure_errors": structure_errors,
                "file_size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


class AtomicFileWriter:
    """Provides atomic file write operations to prevent corruption"""
    
    def __init__(self, target_path: Path):
        self.target_path = target_path
        self.temp_path: Optional[Path] = None
        self._temp_fd: Optional[int] = None
    
    @asynccontextmanager
    async def write_context(self):
        """Context manager for atomic file writing"""
        try:
            # Create temporary file in same directory
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=self.target_path.parent,
                prefix=f".{self.target_path.name}.tmp"
            )
            self.temp_path = Path(temp_path_str)
            self._temp_fd = temp_fd
            
            # Yield file object for writing
            async with aiofiles.open(temp_fd, 'w', encoding='utf-8', closefd=False) as f:
                yield f
            
            # Sync to disk before moving
            os.fsync(temp_fd)
            os.close(temp_fd)
            self._temp_fd = None
            
            # Atomic move to target location
            self.temp_path.replace(self.target_path)
            self.temp_path = None
            
            logger.debug(f"Atomically wrote file: {self.target_path}")
            
        except Exception as e:
            # Clean up temporary file on error
            if self._temp_fd is not None:
                try:
                    os.close(self._temp_fd)
                except Exception:
                    pass
                self._temp_fd = None
            
            if self.temp_path and self.temp_path.exists():
                try:
                    self.temp_path.unlink()
                except Exception:
                    pass
                self.temp_path = None
            
            raise FilesystemException(
                f"Atomic write failed: {str(e)}",
                error_code=ErrorCode.FS_ACCESS_DENIED,
                context=ErrorContext(
                    operation="atomic_file_write",
                    component="atomic_file_writer",
                    additional_data={"target_path": str(self.target_path)}
                ),
                original_exception=e
            )


class EnhancedFileStore:
    """Enhanced file storage with comprehensive error handling and monitoring"""
    
    def __init__(
        self,
        attempts_path: str,
        findings_path: str,
        reports_dir: str,
        enable_compression: bool = False,
        enable_integrity_checks: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_enabled: bool = True
    ):
        self.attempts_path = Path(attempts_path)
        self.findings_path = Path(findings_path)
        self.reports_dir = Path(reports_dir)
        
        self.enable_compression = enable_compression
        self.enable_integrity_checks = enable_integrity_checks
        self.max_file_size = max_file_size
        self.backup_enabled = backup_enabled
        
        # Create directories
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        self.findings_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file locks
        self._attempts_lock = FileLockManager(str(self.attempts_path))
        self._findings_lock = FileLockManager(str(self.findings_path))
        
        # Statistics
        self._write_count = 0
        self._read_count = 0
        self._error_count = 0
        self._corruption_count = 0
        self._bytes_written = 0
        self._bytes_read = 0
        
        logger.info(f"Initialized enhanced file store: attempts={self.attempts_path}, "
                   f"findings={self.findings_path}, reports={self.reports_dir}")
    
    @handle_exceptions(
        component="file_store",
        operation="append_attempt"
    )
    async def append_attempt(self, attempt_data: Dict[str, Any]):
        """Append attempt data to JSONL file with error handling"""
        await self._append_to_jsonl(self.attempts_path, attempt_data, self._attempts_lock)
    
    @handle_exceptions(
        component="file_store", 
        operation="append_finding"
    )
    async def append_finding(self, finding_data: Dict[str, Any]):
        """Append finding data to JSONL file with error handling"""
        await self._append_to_jsonl(self.findings_path, finding_data, self._findings_lock)
    
    async def _append_to_jsonl(
        self,
        file_path: Path,
        data: Dict[str, Any],
        lock_manager: FileLockManager
    ):
        """Safely append data to JSONL file"""
        start_time = time.time()
        
        try:
            # Validate input data
            validation_result = global_validator.validate_dict(data, sanitize=True)
            if not validation_result.is_valid:
                logger.warning(f"Data validation warnings for {file_path}: {validation_result.errors}")
            
            # Use sanitized data if available
            sanitized_data = validation_result.sanitized_data or data
            
            # Serialize to JSON
            json_line = json.dumps(sanitized_data, ensure_ascii=False, separators=(',', ':'))
            json_line += '\n'
            json_bytes = json_line.encode('utf-8')
            
            # Check file size limits
            if file_path.exists():
                current_size = file_path.stat().st_size
                if current_size + len(json_bytes) > self.max_file_size:
                    await self._rotate_file(file_path)
            
            # Acquire file lock and append
            async with lock_manager.acquire_lock():
                # Check integrity before writing if enabled
                original_checksum = None
                if self.enable_integrity_checks and file_path.exists():
                    original_checksum = FileIntegrityChecker.calculate_file_checksum(file_path)
                
                # Append data atomically
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                    await f.write(json_line)
                    await f.flush()
                    await asyncio.get_event_loop().run_in_executor(
                        None, os.fsync, f.fileno()
                    )
                
                # Verify integrity after writing
                if self.enable_integrity_checks:
                    integrity_result = await FileIntegrityChecker.validate_file_integrity(file_path)
                    if not integrity_result["valid"]:
                        self._corruption_count += 1
                        raise FileCorruptionError(
                            f"File corruption detected after write: {integrity_result}",
                            context=ErrorContext(
                                operation="integrity_check_post_write",
                                component="file_store",
                                additional_data={
                                    "file_path": str(file_path),
                                    "integrity_result": integrity_result
                                }
                            )
                        )
            
            # Update statistics
            self._write_count += 1
            self._bytes_written += len(json_bytes)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            await add_custom_metric(
                name="file_store.write.duration_ms",
                value=duration_ms,
                metric_type=MetricType.TIMER,
                tags={"file_type": file_path.stem}
            )
            
            await add_custom_metric(
                name="file_store.write.bytes",
                value=len(json_bytes),
                metric_type=MetricType.COUNTER,
                tags={"file_type": file_path.stem}
            )
            
            logger.debug(f"Appended {len(json_bytes)} bytes to {file_path} in {duration_ms:.1f}ms")
            
        except Exception as e:
            self._error_count += 1
            
            await add_custom_metric(
                name="file_store.write.error",
                value=1,
                metric_type=MetricType.COUNTER,
                tags={"error_type": type(e).__name__, "file_type": file_path.stem}
            )
            
            if not isinstance(e, BaseRedTeamException):
                error = FilesystemException(
                    f"Failed to append to {file_path}: {str(e)}",
                    error_code=ErrorCode.FS_ACCESS_DENIED,
                    context=ErrorContext(
                        operation="jsonl_append",
                        component="file_store",
                        additional_data={"file_path": str(file_path)}
                    ),
                    original_exception=e
                )
                await error_tracker.track_error(error)
                raise error
            else:
                raise
    
    @handle_exceptions(
        component="file_store",
        operation="load_attempts"
    )
    async def load_attempts(self) -> List[Dict[str, Any]]:
        """Load all attempts with error recovery"""
        return await self._load_jsonl(self.attempts_path, self._attempts_lock)
    
    @handle_exceptions(
        component="file_store",
        operation="load_findings"
    )
    async def load_findings(self) -> List[Dict[str, Any]]:
        """Load all findings with error recovery"""
        return await self._load_jsonl(self.findings_path, self._findings_lock)
    
    async def _load_jsonl(
        self,
        file_path: Path,
        lock_manager: FileLockManager
    ) -> List[Dict[str, Any]]:
        """Safely load data from JSONL file with corruption recovery"""
        start_time = time.time()
        
        if not file_path.exists():
            return []
        
        try:
            records = []
            corrupted_lines = []
            
            async with lock_manager.acquire_lock():
                # Check file integrity first
                if self.enable_integrity_checks:
                    integrity_result = await FileIntegrityChecker.validate_file_integrity(file_path)
                    if not integrity_result["valid"]:
                        logger.warning(f"File integrity issues detected: {integrity_result}")
                        
                        # Attempt recovery if there are structure errors
                        if integrity_result.get("structure_errors"):
                            await self._attempt_file_recovery(file_path, integrity_result["structure_errors"])
                
                # Read and parse file
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    line_number = 0
                    async for line in f:
                        line_number += 1
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            corrupted_lines.append({
                                "line_number": line_number,
                                "content": line[:100] + "..." if len(line) > 100 else line,
                                "error": str(e)
                            })
                            logger.warning(f"Skipping corrupted line {line_number} in {file_path}: {e}")
            
            # Update statistics
            self._read_count += 1
            if file_path.exists():
                self._bytes_read += file_path.stat().st_size
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            await add_custom_metric(
                name="file_store.read.duration_ms",
                value=duration_ms,
                metric_type=MetricType.TIMER,
                tags={"file_type": file_path.stem}
            )
            
            await add_custom_metric(
                name="file_store.read.records",
                value=len(records),
                metric_type=MetricType.GAUGE,
                tags={"file_type": file_path.stem}
            )
            
            if corrupted_lines:
                await add_custom_metric(
                    name="file_store.read.corrupted_lines",
                    value=len(corrupted_lines),
                    metric_type=MetricType.COUNTER,
                    tags={"file_type": file_path.stem}
                )
                
                logger.warning(f"Found {len(corrupted_lines)} corrupted lines in {file_path}")
            
            logger.debug(f"Loaded {len(records)} records from {file_path} in {duration_ms:.1f}ms")
            
            return records
            
        except Exception as e:
            self._error_count += 1
            
            await add_custom_metric(
                name="file_store.read.error",
                value=1,
                metric_type=MetricType.COUNTER,
                tags={"error_type": type(e).__name__, "file_type": file_path.stem}
            )
            
            if not isinstance(e, BaseRedTeamException):
                error = FilesystemException(
                    f"Failed to load {file_path}: {str(e)}",
                    error_code=ErrorCode.FS_ACCESS_DENIED,
                    context=ErrorContext(
                        operation="jsonl_load",
                        component="file_store",
                        additional_data={"file_path": str(file_path)}
                    ),
                    original_exception=e
                )
                await error_tracker.track_error(error)
                raise error
            else:
                raise
    
    async def _attempt_file_recovery(self, file_path: Path, structure_errors: List[str]):
        """Attempt to recover corrupted JSONL file"""
        if not self.backup_enabled:
            return
        
        try:
            logger.info(f"Attempting recovery of {file_path} with {len(structure_errors)} errors")
            
            # Create backup of corrupted file
            backup_path = file_path.with_suffix(f"{file_path.suffix}.corrupted.{int(time.time())}")
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup of corrupted file: {backup_path}")
            
            # Read and filter valid lines
            valid_lines = []
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                async for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        json.loads(line)  # Validate JSON
                        valid_lines.append(line)
                    except json.JSONDecodeError:
                        logger.debug(f"Skipping invalid line {line_number} during recovery")
            
            # Write recovered data atomically
            async with AtomicFileWriter(file_path).write_context() as f:
                for line in valid_lines:
                    await f.write(line + '\n')
            
            logger.info(f"Recovered {len(valid_lines)} valid lines from {file_path}")
            
        except Exception as e:
            logger.error(f"File recovery failed for {file_path}: {e}")
            raise
    
    async def _rotate_file(self, file_path: Path):
        """Rotate file when it gets too large"""
        if not file_path.exists():
            return
        
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            rotated_path = file_path.with_suffix(f".{timestamp}{file_path.suffix}")
            
            shutil.move(str(file_path), str(rotated_path))
            logger.info(f"Rotated large file: {file_path} -> {rotated_path}")
            
            await add_custom_metric(
                name="file_store.file_rotation",
                value=1,
                metric_type=MetricType.COUNTER,
                tags={"file_type": file_path.stem}
            )
            
        except Exception as e:
            logger.error(f"File rotation failed for {file_path}: {e}")
            raise
    
    @handle_exceptions(
        component="file_store",
        operation="save_report"
    )
    async def save_report(self, report_name: str, content: str) -> str:
        """Save report with atomic write operations"""
        try:
            # Validate report name
            safe_name = "".join(c for c in report_name if c.isalnum() or c in "._-")
            if not safe_name:
                safe_name = f"report_{int(time.time())}"
            
            report_path = self.reports_dir / f"{safe_name}.md"
            
            # Validate content
            validation_result = global_validator.validate_field("user_input", content)
            if not validation_result.is_valid:
                logger.warning(f"Report content validation warnings: {validation_result.errors}")
            
            sanitized_content = validation_result.sanitized_data or content
            
            # Write atomically
            async with AtomicFileWriter(report_path).write_context() as f:
                await f.write(sanitized_content)
            
            logger.info(f"Saved report: {report_path}")
            
            await add_custom_metric(
                name="file_store.report.saved",
                value=1,
                metric_type=MetricType.COUNTER
            )
            
            return str(report_path)
            
        except Exception as e:
            if not isinstance(e, BaseRedTeamException):
                error = FilesystemException(
                    f"Failed to save report {report_name}: {str(e)}",
                    error_code=ErrorCode.FS_ACCESS_DENIED,
                    original_exception=e
                )
                await error_tracker.track_error(error)
                raise error
            else:
                raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive file store statistics"""
        stats = {
            "operations": {
                "write_count": self._write_count,
                "read_count": self._read_count,
                "error_count": self._error_count,
                "corruption_count": self._corruption_count
            },
            "bytes": {
                "bytes_written": self._bytes_written,
                "bytes_read": self._bytes_read
            },
            "configuration": {
                "enable_compression": self.enable_compression,
                "enable_integrity_checks": self.enable_integrity_checks,
                "max_file_size": self.max_file_size,
                "backup_enabled": self.backup_enabled
            },
            "files": {}
        }
        
        # Add file-specific stats
        for name, path in [
            ("attempts", self.attempts_path),
            ("findings", self.findings_path)
        ]:
            if path.exists():
                stat = path.stat()
                integrity = await FileIntegrityChecker.validate_file_integrity(path)
                
                stats["files"][name] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "integrity_valid": integrity["valid"],
                    "structure_errors": len(integrity.get("structure_errors", []))
                }
            else:
                stats["files"][name] = {"exists": False}
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "healthy": True,
                "checks": {},
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            # Check file accessibility
            for name, path in [
                ("attempts", self.attempts_path),
                ("findings", self.findings_path),
                ("reports_dir", self.reports_dir)
            ]:
                if path.exists():
                    readable = os.access(path, os.R_OK)
                    writable = os.access(path, os.W_OK)
                    
                    health_status["checks"][name] = {
                        "exists": True,
                        "readable": readable,
                        "writable": writable,
                        "healthy": readable and writable
                    }
                    
                    if not (readable and writable):
                        health_status["healthy"] = False
                else:
                    # Check if parent directory is writable
                    parent_writable = os.access(path.parent, os.W_OK)
                    health_status["checks"][name] = {
                        "exists": False,
                        "parent_writable": parent_writable,
                        "healthy": parent_writable
                    }
                    
                    if not parent_writable:
                        health_status["healthy"] = False
            
            # Check disk space
            statvfs = os.statvfs(self.attempts_path.parent)
            free_bytes = statvfs.f_frsize * statvfs.f_available
            total_bytes = statvfs.f_frsize * statvfs.f_blocks
            free_percent = (free_bytes / total_bytes) * 100
            
            health_status["disk_space"] = {
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "free_percent": free_percent,
                "healthy": free_percent > 5.0  # At least 5% free
            }
            
            if free_percent <= 5.0:
                health_status["healthy"] = False
            
            return health_status
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(UTC).isoformat()
            }


# Backwards compatibility
FileStore = EnhancedFileStore


def read_jsonl_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Synchronous convenience function to read JSONL records"""
    path = Path(file_path)
    if not path.exists():
        return []
    
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping corrupted line {line_num} in {path}: {e}")
    
    return records


def append_jsonl(file_path: Union[str, Path], data: Dict[str, Any]):
    """Synchronous convenience function to append JSONL record"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    json_line = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json_line)
        f.flush()
        os.fsync(f.fileno())