#!/usr/bin/env python3
# ABOUTME: End-to-end test script for running 100+ attempts against Llama-8B
# ABOUTME: Validates the full pipeline with batch processing and cost tracking

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from apps.runner.app.main import app
from apps.runner.app.orchestrator import AsyncOrchestrator
from apps.runner.app.providers.openrouter import OpenRouterClient
from apps.runner.app.store.db import StateDAO
from apps.runner.app.util.config import load_config
from apps.runner.app.monitoring.cost_tracker import CostTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "target_model": "meta-llama/llama-3.1-8b-instruct",  # Using Llama-8B for testing
    "num_attempts": 100,  # Number of attempts to run
    "max_concurrency": 5,  # Concurrent requests
    "cost_cap_usd": 1.0,  # Conservative cost cap for testing
    "categories": [
        "harmful_content",
        "system_prompts", 
        "privacy",
        "misinformation",
        "malicious_use"
    ],
    "batch_size": 10,  # Process in batches of 10
    "timeout_seconds": 600  # 10 minute timeout
}


async def initialize_components():
    """Initialize all components for E2E test"""
    # Load environment and config
    load_dotenv()
    config = load_config()
    
    # Verify API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "sk-or-your-api-key-here":
        logger.error("Please set a valid OPENROUTER_API_KEY in .env file")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Initialize database
    dao = StateDAO("data/e2e_test.db")
    await dao.initialize()
    
    # Initialize OpenRouter client
    or_client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
        max_retries=3
    )
    
    # Initialize cost tracker
    cost_tracker = CostTracker(
        cost_cap_usd=TEST_CONFIG["cost_cap_usd"],
        alert_thresholds=[0.25, 0.5, 0.75, 0.9]
    )
    cost_tracker.set_checkpoint_file("data/e2e_cost_checkpoint.json")
    
    # Initialize orchestrator
    orchestrator = AsyncOrchestrator(
        dao=dao,
        openrouter_client=or_client,
        config=config,
        cost_tracker=cost_tracker
    )
    
    return orchestrator, cost_tracker, dao


async def run_e2e_test():
    """Run end-to-end test with 100+ attempts"""
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("Starting E2E Test against Llama-8B")
    logger.info(f"Configuration: {json.dumps(TEST_CONFIG, indent=2)}")
    logger.info("="*60)
    
    try:
        # Initialize components
        orchestrator, cost_tracker, dao = await initialize_components()
        
        # Create a test run
        run_id = f"e2e_test_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Created test run: {run_id}")
        
        # Start the run
        run_metadata = {
            "run_id": run_id,
            "target_model": TEST_CONFIG["target_model"],
            "num_attempts": TEST_CONFIG["num_attempts"],
            "categories": TEST_CONFIG["categories"],
            "test_type": "e2e_llama8b",
            "started_at": datetime.now(UTC).isoformat()
        }
        
        await dao.upsert_state("CURRENT_RUN", run_metadata)
        
        # Run the orchestrator
        logger.info(f"Starting orchestrator for {TEST_CONFIG['num_attempts']} attempts...")
        
        # Create run task with timeout
        run_task = asyncio.create_task(
            orchestrator.run(
                run_id=run_id,
                target_model=TEST_CONFIG["target_model"],
                categories=TEST_CONFIG["categories"],
                max_attempts=TEST_CONFIG["num_attempts"],
                max_concurrency=TEST_CONFIG["max_concurrency"]
            )
        )
        
        # Monitor progress
        monitor_task = asyncio.create_task(
            monitor_progress(orchestrator, cost_tracker)
        )
        
        # Wait for completion or timeout
        try:
            await asyncio.wait_for(run_task, timeout=TEST_CONFIG["timeout_seconds"])
            logger.info("Run completed successfully!")
        except asyncio.TimeoutError:
            logger.warning(f"Run timed out after {TEST_CONFIG['timeout_seconds']} seconds")
            run_task.cancel()
        
        # Cancel monitoring
        monitor_task.cancel()
        
        # Get final statistics
        await print_final_stats(orchestrator, cost_tracker, dao, run_id)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total test time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"E2E test failed: {e}", exc_info=True)
        sys.exit(1)


async def monitor_progress(orchestrator, cost_tracker):
    """Monitor and report progress periodically"""
    while True:
        try:
            await asyncio.sleep(10)  # Report every 10 seconds
            
            # Get current stats
            stats = orchestrator.get_status()
            cost_metrics = cost_tracker.get_metrics()
            
            logger.info(
                f"Progress: "
                f"Attempts: {stats.get('completed_attempts', 0)}/{TEST_CONFIG['num_attempts']} | "
                f"Cost: ${cost_metrics.total_cost:.4f}/${TEST_CONFIG['cost_cap_usd']} | "
                f"Rate: {cost_metrics.cost_per_hour:.4f}/hr | "
                f"Cache hits: {stats.get('cache_hits', 0)} | "
                f"Findings: {stats.get('novel_findings', 0)}"
            )
            
            # Check if we should stop due to cost
            if cost_tracker.should_stop():
                logger.warning("Cost cap reached - stopping test")
                break
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")


async def print_final_stats(orchestrator, cost_tracker, dao, run_id):
    """Print comprehensive final statistics"""
    logger.info("="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    
    # Get orchestrator stats
    orch_stats = orchestrator.get_status()
    
    # Get cost stats
    cost_report = cost_tracker.get_spending_report()
    
    # Get findings
    findings = await dao.get_findings(run_id, limit=10)
    
    # Print summary
    print("\n📊 EXECUTION SUMMARY")
    print(f"  • Run ID: {run_id}")
    print(f"  • Target Model: {TEST_CONFIG['target_model']}")
    print(f"  • Total Attempts: {orch_stats.get('completed_attempts', 0)}/{TEST_CONFIG['num_attempts']}")
    print(f"  • Success Rate: {orch_stats.get('success_rate', 0):.1%}")
    print(f"  • Novel Findings: {len(findings)}")
    
    print("\n💰 COST ANALYSIS")
    print(f"  • Total Cost: ${cost_report['summary']['total_cost_usd']:.4f}")
    print(f"  • Cost Cap: ${cost_report['summary']['cost_cap_usd']:.2f}")
    print(f"  • Usage: {cost_report['summary']['percentage_used']:.1f}%")
    print(f"  • Avg per Request: ${cost_report['rates']['average_cost_per_request_usd']:.6f}")
    print(f"  • Tokens Used: {cost_report['rates']['tokens_used']:,}")
    
    print("\n⚡ PERFORMANCE METRICS")
    if 'evaluator' in orch_stats and orch_stats['evaluator']:
        eval_stats = orch_stats['evaluator']
        print(f"  • Cache Hit Rate: {eval_stats.get('cache_hit_rate_percent', 0):.1f}%")
        print(f"  • Judge Escalation Rate: {eval_stats.get('escalation_rate_percent', 0):.1f}%")
        print(f"  • Avg Evaluation Time: {eval_stats.get('avg_evaluation_time_ms', 0):.0f}ms")
        print(f"  • Cost Savings: {eval_stats.get('estimated_cost_savings_percent', 0):.1f}%")
    
    print("\n🎯 CATEGORY BREAKDOWN")
    if 'bandit' in orch_stats and orch_stats['bandit']:
        bandit_stats = orch_stats['bandit'].get('category_stats', {})
        for category, stats in bandit_stats.items():
            print(f"  • {category}: {stats.get('attempts', 0)} attempts, "
                  f"{stats.get('successes', 0)} successes, "
                  f"score: {stats.get('success_rate', 0):.3f}")
    
    print("\n🔍 TOP FINDINGS")
    for i, finding in enumerate(findings[:5], 1):
        print(f"  {i}. Category: {finding.get('category', 'unknown')}")
        print(f"     Severity: {finding.get('severity', 'unknown')}")
        print(f"     Confidence: {finding.get('confidence', 0):.2f}")
        if 'prompt' in finding:
            prompt_preview = finding['prompt'][:100] + "..." if len(finding['prompt']) > 100 else finding['prompt']
            print(f"     Prompt: {prompt_preview}")
    
    # Save detailed report
    report_path = Path(f"data/e2e_report_{run_id}.json")
    full_report = {
        "test_config": TEST_CONFIG,
        "execution_summary": orch_stats,
        "cost_report": cost_report,
        "findings_count": len(findings),
        "top_findings": [f for f in findings[:10]]
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Check for any warnings
    print("\n⚠️  WARNINGS")
    if cost_report['summary']['percentage_used'] > 90:
        print("  • Cost usage exceeded 90% of cap")
    if orch_stats.get('completed_attempts', 0) < TEST_CONFIG['num_attempts']:
        print(f"  • Only completed {orch_stats.get('completed_attempts', 0)}/{TEST_CONFIG['num_attempts']} attempts")
    if len(findings) == 0:
        print("  • No novel findings discovered")


if __name__ == "__main__":
    print("\n🚀 OpenAI Red Team E2E Test Runner")
    print("   Testing with Llama-8B model")
    print("="*60)
    
    # Run the test
    asyncio.run(run_e2e_test())