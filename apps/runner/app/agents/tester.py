# ABOUTME: Tester agent that executes prompts against target models and records results
# ABOUTME: Handles error classification, usage tracking, and JSONL transcript generation

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.runner.app.providers.ollama import call_ollama
from apps.runner.app.store.files import append_jsonl, redact_pii, rotate_file_if_needed
from apps.runner.app.util.cost import estimate_cost


def classify_error(error_info: Optional[Dict[str, Any]]) -> str:
    """
    Classify an error as temporary (retryable) or permanent.
    
    Args:
        error_info: Error information dictionary with status code and message
        
    Returns:
        'temporary' or 'permanent'
    """
    if not error_info:
        return "temporary"
    
    status = error_info.get("status", 0)
    
    # Temporary errors (should retry)
    if status in [429, 500, 502, 503, 504]:
        return "temporary"
    
    # Permanent errors (should not retry)
    if status in [400, 401, 403, 404, 405, 422]:
        return "permanent"
    
    # Default to temporary for unknown errors
    return "temporary"


async def run_attempt(
    prompt: str,
    model: str,
    jsonl_path: Path,
    template_id: str,
    mutator_chain: List[str],
    seed: int,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute a single attempt against the target model.
    
    Args:
        prompt: The crafted prompt to send
        model: Model identifier (e.g., "meta-llama/llama-3.1-8b-instruct")
        jsonl_path: Path to append the attempt record
        template_id: ID of the template used
        mutator_chain: List of mutators applied
        seed: Random seed used for determinism
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with attempt results including success status, response, usage, and errors
    """
    attempt_id = f"attempt-{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Prepare the record
    record: Dict[str, Any] = {
        "id": attempt_id,
        "timestamp": timestamp,
        "prompt": redact_pii(prompt),  # Redact PII from prompt
        "model": model,
        "response": None,
        "usage": None,
        "cost_usd": 0.0,
        "error": None,
        "metadata": {
            "template_id": template_id,
            "mutator_chain": mutator_chain,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }
    
    result = {
        "success": False,
        "attempt_id": attempt_id,
        "response": None,
        "usage": {},
        "cost_usd": 0.0,
        "error": None
    }
    
    try:
        # Check for file rotation before writing
        rotate_file_if_needed(jsonl_path, max_size_mb=100)
        
        # Call the model via Ollama
        messages = [{"role": "user", "content": prompt}]
        response, usage = await asyncio.wait_for(
            call_ollama(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            ),
            timeout=timeout
        )
        
        # Redact PII from response
        response = redact_pii(response)
        
        # Calculate cost
        cost_usd = estimate_cost(model, usage)
        
        # Update record with success
        record["response"] = response
        record["usage"] = usage
        record["cost_usd"] = cost_usd
        
        # Update result
        result["success"] = True
        result["response"] = response
        result["usage"] = usage  # This is the full usage dict
        result["cost_usd"] = cost_usd
        
    except asyncio.TimeoutError:
        error_info = {
            "type": "timeout",
            "message": f"Request timed out after {timeout} seconds",
            "status": 504
        }
        record["error"] = error_info
        result["error"] = error_info
        
    except Exception as e:
        # Extract error details
        error_message = str(e)
        error_status = 500  # Default status
        
        # Try to extract status code from error message
        if "429" in error_message:
            error_status = 429
        elif "401" in error_message or "unauthorized" in error_message.lower():
            error_status = 401
        elif "404" in error_message:
            error_status = 404
        elif "400" in error_message or "invalid" in error_message.lower():
            error_status = 400
        
        error_info = {
            "type": "api_error",
            "message": error_message,
            "status": error_status,
            "classification": classify_error({"status": error_status})
        }
        
        record["error"] = error_info
        result["error"] = error_info
    
    finally:
        # Always append the record to JSONL
        append_jsonl(jsonl_path, record)
    
    return result


async def run_batch_attempts(
    attempts: List[Dict[str, Any]],
    model: str,
    jsonl_path: Path,
    max_concurrency: int = 5,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run multiple attempts with controlled concurrency.
    
    Args:
        attempts: List of attempt configurations
        model: Target model identifier
        jsonl_path: Path to JSONL file for results
        max_concurrency: Maximum concurrent requests
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of attempt results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore(attempt_config):
        async with semaphore:
            result = await run_attempt(
                prompt=attempt_config["prompt"],
                model=model,
                jsonl_path=jsonl_path,
                template_id=attempt_config.get("template_id", "unknown"),
                mutator_chain=attempt_config.get("mutator_chain", []),
                seed=attempt_config.get("seed", 0),
                max_tokens=attempt_config.get("max_tokens", 1000),
                temperature=attempt_config.get("temperature", 0.7)
            )
            
            if progress_callback:
                progress_callback(result)
            
            return result
    
    tasks = [run_with_semaphore(attempt) for attempt in attempts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that made it through
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_result = {
                "success": False,
                "attempt_id": f"error-{i}",
                "response": None,
                "usage": None,
                "cost_usd": 0.0,
                "error": {
                    "type": "batch_error",
                    "message": str(result),
                    "status": 500
                }
            }
            processed_results.append(error_result)
        else:
            processed_results.append(result)
    
    return processed_results


class AttemptRunner:
    """High-level interface for running attempts with state management."""
    
    def __init__(self, model: str, jsonl_path: Path, max_concurrency: int = 5):
        self.model = model
        self.jsonl_path = Path(jsonl_path)
        self.max_concurrency = max_concurrency
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_cost = 0.0
    
    async def run_single(
        self,
        prompt: str,
        template_id: str = "manual",
        mutator_chain: List[str] = None,
        seed: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a single attempt and update statistics."""
        result = await run_attempt(
            prompt=prompt,
            model=self.model,
            jsonl_path=self.jsonl_path,
            template_id=template_id,
            mutator_chain=mutator_chain or [],
            seed=seed,
            **kwargs
        )
        
        self.total_attempts += 1
        if result["success"]:
            self.successful_attempts += 1
            self.total_cost += result.get("cost_usd", 0.0)
        else:
            self.failed_attempts += 1
        
        return result
    
    async def run_batch(self, attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a batch of attempts with progress tracking."""
        def progress_update(result):
            self.total_attempts += 1
            if result["success"]:
                self.successful_attempts += 1
                self.total_cost += result.get("cost_usd", 0.0)
            else:
                self.failed_attempts += 1
        
        results = await run_batch_attempts(
            attempts=attempts,
            model=self.model,
            jsonl_path=self.jsonl_path,
            max_concurrency=self.max_concurrency,
            progress_callback=progress_update
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": (
                self.successful_attempts / self.total_attempts 
                if self.total_attempts > 0 else 0
            ),
            "total_cost_usd": self.total_cost
        }