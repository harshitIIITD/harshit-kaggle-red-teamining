# ABOUTME: Fully async orchestrator using aiosqlite for all database operations
# ABOUTME: Eliminates threading issues by keeping everything in the main event loop

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict
import random
import uuid

from apps.runner.app.store.async_db import AsyncDatabasePool, get_state, set_state
from apps.runner.app.providers.openrouter import OpenRouterClient
from apps.runner.app.agents.planner import Planner
from apps.runner.app.agents.crafter import PromptCrafter
from apps.runner.app.agents.tester import run_attempt
from apps.runner.app.agents.evaluator import EvaluationOrchestrator as Evaluator
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaskItem:
    """Task item for scheduling"""
    id: str
    category: str
    strategy: str
    template_id: str
    priority: float
    metadata: Dict[str, Any]
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority > other.priority


class CircuitBreaker:
    """Circuit breaker for error rate management"""
    
    def __init__(self, threshold: float = 0.5, window_size: int = 10, cooldown_seconds: int = 30):
        self.threshold = threshold
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self.errors = []
        self.last_open_time = None
    
    def record_success(self):
        """Record a successful execution"""
        self.errors.append(0)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
    
    def record_failure(self):
        """Record a failed execution"""
        self.errors.append(1)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open (tripped)"""
        if self.last_open_time:
            # Check if cooldown period has passed
            if time.time() - self.last_open_time > self.cooldown_seconds:
                self.last_open_time = None
                self.errors.clear()
                return False
            return True
        
        # Check error rate
        if len(self.errors) >= self.window_size:
            error_rate = sum(self.errors) / len(self.errors)
            if error_rate > self.threshold:
                self.last_open_time = time.time()
                logger.warning(f"Circuit breaker tripped! Error rate: {error_rate:.2%}")
                return True
        
        return False


class ThompsonSampling:
    """Thompson Sampling bandit algorithm using Beta distribution"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.arms = defaultdict(lambda: {"alpha": alpha, "beta": beta, "successes": 0, "failures": 0})
    
    def select_arm(self, arms: List[str]) -> str:
        """Select arm using Thompson sampling"""
        if not arms:
            raise ValueError("No arms to select from")
        
        # Sample from Beta distribution for each arm
        samples = {}
        for arm in arms:
            arm_stats = self.arms[arm]
            # Sample from Beta(alpha + successes, beta + failures)
            sample = random.betavariate(
                arm_stats["alpha"] + arm_stats["successes"],
                arm_stats["beta"] + arm_stats["failures"]
            )
            samples[arm] = sample
        
        # Select arm with highest sample
        return max(samples, key=samples.get)
    
    def update(self, arm: str, reward: float) -> None:
        """Update arm statistics with observed reward"""
        if reward > 0.5:  # Success threshold
            self.arms[arm]["successes"] += 1
        else:
            self.arms[arm]["failures"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for all arms"""
        return dict(self.arms)


class AsyncOrchestrator:
    """Fully async orchestrator managing task execution with bandit algorithms"""
    
    def __init__(self, config: Dict[str, Any], db_pool: AsyncDatabasePool):
        """
        Initialize async orchestrator
        
        Args:
            config: Configuration dictionary
            db_pool: Async database connection pool
        """
        self.config = config
        self.db_pool = db_pool
        
        # Parse config
        run_config = config.get("run", {})
        self.max_concurrency = run_config.get("max_concurrency", 6)
        self.cost_cap_usd = run_config.get("cost_cap_usd", 10.0)
        self.categories = run_config.get("categories", ["safety", "privacy", "fairness"])
        self.strategies = ["direct", "roleplay", "encoding"]  # Fixed strategies for now
        
        # Initialize bandit policy
        self.bandit_policy = ThompsonSampling()
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=0.5,
            window_size=10,
            cooldown_seconds=30
        )
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = set()
        
        # Metrics
        self.attempts_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_cost = 0.0
        
        # Run identifier
        self.run_id = None
        self.max_attempts = 1000
        
        # Agents and services
        self.openrouter_client = None
        self.crafter = None
        self.tester = None
        self.evaluator = None
        self.transcript_file = None
        
        # Control state
        self._shutdown = False
    
    async def initialize(self) -> None:
        """Initialize agents and services"""
        # Initialize OpenRouter client with config parameters
        provider_config = self.config.get("providers", {}).get("openrouter", {})
        self.openrouter_client = OpenRouterClient(
            base_url=provider_config.get("base_url", "https://openrouter.ai/api/v1"),
            timeout=provider_config.get("timeout", 60)
            # api_key will be read from env var by default
        )
        
        # Initialize planner
        self.planner = Planner(config=self.config.get("planner", {}))
        logger.info("Planner initialized")
        
        # Initialize transcript file path
        self.transcript_file = self.config["storage"]["transcripts_path"]
        
        # Generate initial tasks
        await self._generate_initial_tasks()
        
        logger.info("Async orchestrator initialized")
    
    async def _generate_initial_tasks(self) -> None:
        """Generate initial task backlog using Planner"""
        # Use planner to generate backlog
        enabled_categories = self.config.get("enabled_categories", self.categories)
        arms_per_category = self.config.get("arms_per_category", 5)
        
        if self.planner is None:
            # Fallback to simple generation if planner not initialized
            logger.warning("Planner not initialized, using simple task generation")
            task_count = 0
            for category in enabled_categories:
                for strategy in self.strategies:
                    for i in range(3):
                        task = TaskItem(
                            id=f"{category}-{strategy}-{i}",
                            category=category,
                            strategy=strategy,
                            template_id=f"template-{i}",
                            priority=random.random(),
                            metadata={}
                        )
                        await self.task_queue.put(task)
                        task_count += 1
        else:
            # Use planner to generate arms
            backlog = self.planner.generate_backlog(
                categories=enabled_categories,
                count_per_category=arms_per_category
            )
            
            # Convert Arms to TaskItems and add to queue
            task_count = 0
            for arm in backlog:
                task = TaskItem(
                    id=arm.id,
                    category=arm.category,
                    strategy=arm.strategy_id,
                    template_id=arm.template_id,
                    priority=arm.priority,
                    metadata={
                        "seed": arm.seed,
                        "mutator_chain": arm.mutator_chain,
                        **arm.parameters
                    }
                )
                await self.task_queue.put(task)
                task_count += 1
        
        logger.info(f"Generated {task_count} initial tasks")
    
    async def _get_run_state(self) -> Optional[str]:
        """Get run state from database"""
        async with self.db_pool.acquire() as conn:
            return await get_state(conn, "RUN_STATE")
    
    async def _set_run_state(self, state: str) -> None:
        """Set run state in database"""
        async with self.db_pool.acquire() as conn:
            await set_state(conn, "RUN_STATE", state)
    
    async def checkpoint(self) -> None:
        """Save current state to database"""
        async with self.db_pool.acquire() as conn:
            # Save metrics
            await set_state(conn, "TOTAL_COST", str(self.total_cost))
            await set_state(conn, "ATTEMPTS_COUNT", str(self.attempts_count))
            await set_state(conn, "SUCCESS_COUNT", str(self.success_count))
            await set_state(conn, "ERROR_COUNT", str(self.error_count))
        
        logger.debug("Checkpoint saved")
    
    async def _generate_additional_tasks(self, count: int) -> None:
        """Generate additional tasks when queue is empty but max_attempts not reached"""
        if self.planner is None:
            return
            
        # Use the same categories from config
        enabled_categories = self.config.get("enabled_categories", self.categories)
        
        # Generate a small batch per category
        tasks_per_category = max(1, count // len(enabled_categories))
        
        # Generate new arms
        backlog = self.planner.generate_backlog(
            categories=enabled_categories,
            count_per_category=tasks_per_category
        )
        
        # Add to queue
        for arm in backlog:
            task = TaskItem(
                id=arm.id,
                category=arm.category,
                strategy=arm.strategy,
                template=arm.template_id,
                mutations=arm.mutations,
                seed=arm.seed,
                metadata=arm.metadata
            )
            await self.task_queue.put(task)
        
        logger.info(f"Added {len(backlog)} new tasks to queue")
    
    async def _task_worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown and self.attempts_count < self.max_attempts:
            try:
                # Check if paused
                state = await self._get_run_state()
                if state == "paused":
                    await asyncio.sleep(1)
                    continue
                
                # Check circuit breaker
                if self.circuit_breaker.is_open():
                    logger.warning(f"Worker {worker_id}: Circuit breaker open, pausing")
                    await self._set_run_state("paused")
                    await asyncio.sleep(5)
                    continue
                
                # Check cost cap
                if self.total_cost >= self.cost_cap_usd:
                    logger.warning(f"Cost cap reached: ${self.total_cost:.2f}")
                    self._shutdown = True
                    break
                
                # Get task from queue (with timeout)
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we need to generate more tasks
                    if self.task_queue.empty() and self.attempts_count < self.max_attempts:
                        # Generate additional tasks if we haven't reached max attempts
                        additional_needed = min(10, self.max_attempts - self.attempts_count)
                        if additional_needed > 0 and worker_id == 0:  # Only worker 0 generates
                            logger.info(f"Generating {additional_needed} additional tasks")
                            await self._generate_additional_tasks(additional_needed)
                    continue
                
                # Skip if already completed
                if task.id in self.completed_tasks:
                    continue
                
                # Process task
                logger.info(f"Worker {worker_id} processing task {task.id}")
                self.active_tasks[task.id] = task
                
                try:
                    # Execute task
                    result = await self._execute_task(task)
                    
                    # Update metrics
                    self.attempts_count += 1
                    if result.get("success"):
                        self.success_count += 1
                        self.circuit_breaker.record_success()
                    else:
                        self.error_count += 1
                        self.circuit_breaker.record_failure()
                    
                    self.total_cost += result.get("cost", 0.0)
                    
                    # Update bandit policy
                    arm = f"{task.category}-{task.strategy}"
                    reward = result.get("score", 0.0)
                    self.bandit_policy.update(arm, reward)
                    
                    # Mark as completed
                    self.completed_tasks.add(task.id)
                    
                    # Checkpoint periodically
                    if self.attempts_count % 10 == 0:
                        await self.checkpoint()
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} task {task.id} failed: {e}")
                    self.error_count += 1
                    self.circuit_breaker.record_failure()
                finally:
                    self.active_tasks.pop(task.id, None)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: TaskItem) -> Dict[str, Any]:
        """Execute a single task using real agents"""
        start_time = time.time()
        result = {
            "task_id": task.id,
            "success": False,
            "cost": 0.0,
            "score": 0.0,
            "is_vulnerable": False
        }
        
        try:
            # Use mock mode if configured (for testing)
            use_mock = self.config.get("use_mock_agents", False)
            
            if use_mock:
                # Mock execution for testing
                prompt = f"Test prompt for {task.category} using {task.strategy}"
                await asyncio.sleep(0.1)  # Simulate API delay
                
                test_result = {
                    "response": f"Mock response for task {task.id}",
                    "cost": 0.001,
                    "prompt_tokens": 100,
                    "completion_tokens": 50
                }
                
                eval_result = {
                    "is_vulnerable": random.random() > 0.7,
                    "confidence": random.random(),
                    "severity": "medium",
                    "rationale": "Mock evaluation",
                    "total_cost_usd": 0.0005,
                    "score": random.random()
                }
            else:
                # Real agent execution
                # 1. Craft prompt using PromptCrafter
                crafter = PromptCrafter()
                prompt = crafter.craft(
                    template_id=task.template_id,
                    mutator_chain=task.metadata.get("mutator_chain", ["lexical"]),
                    seed=task.metadata.get("seed", 1234),
                    parameters=task.metadata
                )
                
                # 2. Execute test using Tester agent
                target_model = self.config.get("target_model", "meta-llama/llama-3.1-8b-instruct")
                jsonl_path = Path(self.transcript_file)
                
                test_result = await run_attempt(
                    prompt=prompt,
                    model=target_model,
                    jsonl_path=jsonl_path,
                    template_id=task.template_id,
                    mutator_chain=task.metadata.get("mutator_chain", []),
                    seed=task.metadata.get("seed", 1234),
                    max_tokens=task.metadata.get("max_tokens", 1000),
                    temperature=task.metadata.get("temperature", 0.7)
                )
                
                # 3. Evaluate response using Evaluator
                if self.evaluator is None:
                    eval_config = self.config.get("evaluation", {})
                    judge_model = eval_config.get("judge_model", "meta-llama/llama-3.1-70b-instruct")
                    self.evaluator = Evaluator(
                        openrouter_client=self.openrouter_client,
                        judge_model=judge_model
                    )
                
                from apps.runner.app.util.schemas import Attempt, AttemptStatus, Usage
                from datetime import datetime
                attempt = Attempt(
                    id=f"attempt_{task.id}_{int(time.time())}",
                    task_id=task.id,
                    run_id=self.run_id if hasattr(self, 'run_id') else f"run-{uuid.uuid4().hex[:8]}",
                    status=AttemptStatus.SUCCESS if test_result.get("response") else AttemptStatus.FAILED,
                    prompt=prompt,
                    response=test_result.get("response", ""),
                    usage=Usage(
                        prompt_tokens=test_result.get("prompt_tokens", 0),
                        completion_tokens=test_result.get("completion_tokens", 0),
                        total_tokens=test_result.get("prompt_tokens", 0) + test_result.get("completion_tokens", 0),
                        cost_usd=test_result.get("cost", 0.0)
                    ),
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error_message=test_result.get("error") if not test_result.get("response") else None,
                    metadata={
                        **task.metadata,
                        "model": target_model,
                        "category": task.category,
                        "template_id": task.template_id,
                        "mutator_chain": task.metadata.get("mutator_chain", []),
                        "seed": task.metadata.get("seed", 1234),
                        "latency_ms": test_result.get("latency_ms", 0)
                    }
                )
                
                eval_result = await self.evaluator.evaluate(attempt)
            
            # Update result (eval_result is an Evaluation object)
            result["success"] = True
            result["cost"] = test_result.get("cost", 0.0) + eval_result.total_cost_usd
            result["score"] = eval_result.confidence if eval_result.is_vulnerable else 0.0
            result["is_vulnerable"] = eval_result.is_vulnerable
            result["severity"] = eval_result.severity.value if hasattr(eval_result.severity, 'value') else str(eval_result.severity)
            result["rationale"] = eval_result.rationale
            
            # Check novelty if vulnerability found and detector provided
            if eval_result.is_vulnerable and hasattr(self, 'detector') and self.detector:
                from apps.runner.app.util.schemas import AttemptRecord
                
                # Create attempt record for novelty detection
                attempt_record = AttemptRecord(
                    id=attempt.id,
                    timestamp=datetime.now().isoformat(),
                    prompt=prompt,
                    model=target_model,
                    response=test_result.get("response", ""),
                    usage=test_result.get("usage"),
                    cost_usd=test_result.get("cost", 0.0),
                    error=None,
                    metadata={
                        "category": str(task.category),
                        "template_id": task.template_id,
                        "mutator_chain": task.metadata.get("mutator_chain", [])
                    }
                )
                
                # Process through novelty detector
                cluster_id = self.detector.process_evaluation(eval_result, attempt_record)
                if cluster_id:
                    result["cluster_id"] = cluster_id
                    result["is_novel"] = True
                    logger.info(f"Finding added to cluster {cluster_id} for task {task.id}")
                else:
                    result["is_novel"] = False
            
            logger.info(f"Task {task.id} completed: vulnerable={result['is_vulnerable']}, score={result['score']:.2f}")
            
        except Exception as e:
            logger.error(f"Task execution error: {e}", exc_info=True)
            result["error"] = str(e)
        
        result["duration"] = time.time() - start_time
        return result
    
    async def run(
        self,
        run_id: Optional[str] = None,
        max_attempts: int = 1000,
        detector: Optional[Any] = None,
        reporter: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run the orchestrator
        
        Args:
            run_id: Unique identifier for this run
            max_attempts: Maximum number of attempts to make
            detector: Novelty detector (optional)
            reporter: Report generator (optional)
        
        Returns:
            Dictionary with run results
        """
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.detector = detector  # Store detector for use in execute_task
        self.reporter = reporter  # Store reporter for use later
        self.max_attempts = max_attempts
        
        # Initialize if not already done
        if self.openrouter_client is None:
            await self.initialize()
        
        # Set run state
        await self._set_run_state("running")
        
        logger.info(f"Starting orchestrator run {self.run_id} with {self.max_concurrency} workers")
        
        # Create worker tasks
        workers = [
            asyncio.create_task(self._task_worker(i))
            for i in range(self.max_concurrency)
        ]
        
        start_time = time.time()
        
        try:
            # Wait for all workers to complete
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
        finally:
            self._shutdown = True
            await self._set_run_state("stopped")
            await self.checkpoint()
            
            # Clean up
            if self.openrouter_client:
                await self.openrouter_client.close()
            
            duration = time.time() - start_time
            
            # Get findings from detector if provided
            findings_count = 0
            clusters_count = 0
            if detector:
                findings_count = len(detector.get_findings())
                clusters_count = len(getattr(detector.cluster_store, 'clusters', []))
            
            result = {
                "run_id": self.run_id,
                "status": "completed",
                "total_attempts": self.attempts_count,
                "successful_attempts": self.success_count,
                "failed_attempts": self.error_count,
                "findings_count": findings_count,
                "clusters_count": clusters_count,
                "total_cost": self.total_cost,
                "duration_seconds": duration,
                "bandit_stats": self.bandit_policy.get_stats()
            }
            
            # Generate report if reporter is provided
            if reporter and detector:
                try:
                    logger.info(f"Generating report for run {self.run_id}")
                    report_path = reporter.generate_report(
                        detector=detector,
                        run_id=self.run_id,
                        run_config={
                            "run_id": self.run_id,
                            "target_model": self.config.get("target_model", "unknown"),
                            "total_attempts": self.attempts_count,
                            "successful_evaluations": self.success_count,
                            "unique_clusters": clusters_count,
                            "total_cost_usd": self.total_cost,
                            "duration_seconds": duration,
                            "model": self.config.get("target_model", "unknown"),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                        }
                    )
                    result["report_path"] = report_path
                    logger.info(f"Report generated: {report_path}")
                except Exception as e:
                    logger.error(f"Failed to generate report: {e}")
                    result["report_error"] = str(e)
            
            logger.info(f"Run {self.run_id} completed: {result}")
            return result