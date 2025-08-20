# ABOUTME: Adversarial-resistant judge integration for LLM-based vulnerability evaluation
# ABOUTME: Includes input sanitization, circuit breaker protection, and structured prompts

import json
import time
import logging
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass

from apps.runner.app.util.schemas import Attempt, Severity
from apps.runner.app.providers.openrouter import OpenRouterClient
from .sanitizer import InputSanitizer
from .prompts import get_judge_prompt, validate_judge_response

logger = logging.getLogger(__name__)


class JudgeUnavailableError(Exception):
    """Raised when judge evaluation is unavailable due to failures or circuit breaker"""
    pass


class JudgeEvaluationError(Exception):
    """Raised when judge evaluation fails due to API or parsing errors"""
    pass


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation"""
    confidence: float  # 0.0 to 1.0 - judge's confidence in vulnerability assessment
    is_vulnerable: bool  # Judge's determination of vulnerability
    severity: Severity  # Severity level if vulnerable
    rationale: str  # Judge's explanation of decision
    cost_usd: float  # Cost of judge API call
    evaluation_time_ms: float  # Time taken for evaluation
    sanitization_applied: bool  # Whether input sanitization was required
    metadata: Dict[str, Any]  # Additional context


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascade failures from judge API issues.
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing recovery)
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking calls)"""
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state for recovery test")
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker recovered - moving to CLOSED state")
        self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold and self.state == "CLOSED":
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("Circuit breaker failed recovery test - returning to OPEN state")


class JudgeIntegrator:
    """
    Adversarial-resistant judge integration for vulnerability evaluation.
    Includes input sanitization, circuit breaker protection, and error recovery.
    """
    
    def __init__(self, openrouter_client: OpenRouterClient, judge_model: str = "meta-llama/llama-3.1-70b-instruct"):
        """
        Initialize judge integrator.
        
        Args:
            openrouter_client: OpenRouter client for API calls
            judge_model: Model to use for judge evaluations
        """
        self.client = openrouter_client
        self.judge_model = judge_model
        self.sanitizer = InputSanitizer(max_length=7000)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_cost = 0.0
        self.total_time_ms = 0.0
        
        logger.info(f"JudgeIntegrator initialized with model: {judge_model}")
    
    async def evaluate_with_judge(self, attempt: Attempt, category: str) -> JudgeResult:
        """
        Evaluate attempt using LLM judge with full adversarial protection.
        
        Args:
            attempt: Attempt to evaluate
            category: Safety category for evaluation
            
        Returns:
            JudgeResult with evaluation details
            
        Raises:
            JudgeUnavailableError: Circuit breaker is open or repeated failures
            JudgeEvaluationError: API call or parsing failed
        """
        start_time = time.perf_counter()
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            raise JudgeUnavailableError("Circuit breaker is open - judge temporarily unavailable")
        
        try:
            # Stage 1: Input sanitization
            sanitized_prompt = self.sanitizer.sanitize(attempt.prompt or "")
            sanitized_response = self.sanitizer.sanitize(attempt.response or "")
            
            sanitization_applied = (
                self.sanitizer.is_potentially_adversarial(attempt.prompt or "") or
                self.sanitizer.is_potentially_adversarial(attempt.response or "")
            )
            
            if sanitization_applied:
                logger.debug(f"Input sanitization applied for attempt {attempt.id}")
            
            # Stage 2: Construct judge prompt
            judge_prompt = get_judge_prompt(category, sanitized_prompt, sanitized_response)
            
            # Stage 3: Make API call with retry protection
            response_content, usage = await self._make_judge_call(judge_prompt)
            
            # Stage 4: Parse and validate response
            parsed_result = self._parse_judge_response(response_content)
            
            # Stage 5: Create structured result
            evaluation_time_ms = (time.perf_counter() - start_time) * 1000
            cost_usd = usage.get("cost_usd", 0.0)
            
            # Update performance tracking
            self.evaluation_count += 1
            self.total_cost += cost_usd
            self.total_time_ms += evaluation_time_ms
            
            # Record success for circuit breaker
            self.circuit_breaker.record_success()
            
            return JudgeResult(
                confidence=parsed_result["confidence"],
                is_vulnerable=parsed_result["is_vulnerable"],
                severity=Severity(parsed_result["severity"].lower()),
                rationale=parsed_result["rationale"],
                cost_usd=cost_usd,
                evaluation_time_ms=evaluation_time_ms,
                sanitization_applied=sanitization_applied,
                metadata={
                    "judge_model": self.judge_model,
                    "category": category,
                    "attempt_id": attempt.id,
                    "prompt_length": len(attempt.prompt or ""),
                    "response_length": len(attempt.response or "")
                }
            )
            
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Judge evaluation failed after {elapsed_ms:.2f}ms: {e}")
            
            raise JudgeEvaluationError(f"Judge evaluation failed: {str(e)}")
    
    async def _make_judge_call(self, prompt: str) -> tuple:
        """
        Make API call to judge model with error handling.
        
        Args:
            prompt: Formatted judge prompt
            
        Returns:
            Tuple of (response_content, usage_dict)
            
        Raises:
            JudgeEvaluationError: API call failed
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response, usage = await self.client.chat(
                model=self.judge_model,
                messages=messages,
                max_tokens=200,  # Keep responses concise for cost control
                temperature=0.1,  # Low temperature for consistent evaluation
                timeout=30  # Reasonable timeout for competition environment
            )
            
            return response, usage
            
        except asyncio.TimeoutError:
            raise JudgeEvaluationError("Judge API call timed out")
        except Exception as e:
            raise JudgeEvaluationError(f"Judge API call failed: {str(e)}")
    
    def _parse_judge_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse and validate judge response JSON with error recovery.
        
        Args:
            response_content: Raw response from judge model
            
        Returns:
            Parsed and validated response dictionary
            
        Raises:
            JudgeEvaluationError: Response parsing/validation failed
        """
        if not response_content or not response_content.strip():
            raise JudgeEvaluationError("Empty response from judge model")
        
        # Try to parse JSON - first attempt at full response
        try:
            parsed = json.loads(response_content.strip())
        except json.JSONDecodeError:
            # Recovery attempt: extract JSON from response
            parsed = self._extract_json_from_response(response_content)
            if not parsed:
                raise JudgeEvaluationError(f"Could not parse JSON from response: {response_content[:200]}...")
        
        # Validate response structure
        if not validate_judge_response(parsed):
            raise JudgeEvaluationError(f"Invalid judge response structure: {parsed}")
        
        return parsed
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract JSON from malformed response using regex.
        This handles cases where the judge includes extra text around the JSON.
        """
        import re
        
        # Look for JSON-like structure in the response
        json_pattern = r'\{[^{}]*"is_vulnerable"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if validate_judge_response(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        if self.evaluation_count == 0:
            return {"message": "No judge evaluations performed yet"}
        
        return {
            "total_evaluations": self.evaluation_count,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_evaluation": self.total_cost / self.evaluation_count,
            "avg_time_ms": self.total_time_ms / self.evaluation_count,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }