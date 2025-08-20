#!/usr/bin/env python3
# ABOUTME: E2E test using the FastAPI endpoints to run 100+ attempts
# ABOUTME: Tests the full pipeline through the HTTP API

import asyncio
import httpx
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "target_model": "meta-llama/llama-3.1-8b-instruct",  # Llama-8B for testing
    "num_attempts": 100,  # Number of attempts
    "categories": [
        "harmful_content",
        "system_prompts",
        "privacy",
        "misinformation",
        "malicious_use"
    ],
    "max_concurrency": 5,
    "cost_cap_usd": 1.0,
    "api_base_url": "http://localhost:8000"
}


async def check_server_health():
    """Check if the server is running"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{TEST_CONFIG['api_base_url']}/health")
            if resp.status_code == 200:
                logger.info("✅ Server is healthy")
                return True
            else:
                logger.error(f"❌ Server health check failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to server: {e}")
            logger.info("Please start the server with: uv run uvicorn apps.runner.app.main:app --reload")
            return False


async def start_run():
    """Start a new run via API"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "target_model": TEST_CONFIG["target_model"],
            "categories": TEST_CONFIG["categories"],
            "max_attempts": TEST_CONFIG["num_attempts"],
            "max_concurrency": TEST_CONFIG["max_concurrency"],
            "cost_cap_usd": TEST_CONFIG["cost_cap_usd"]
        }
        
        logger.info(f"Starting run with config: {json.dumps(payload, indent=2)}")
        
        try:
            resp = await client.post(
                f"{TEST_CONFIG['api_base_url']}/runs",
                json=payload
            )
            
            if resp.status_code == 200:
                result = resp.json()
                run_id = result.get("run_id")
                logger.info(f"✅ Run started: {run_id}")
                return run_id
            else:
                logger.error(f"❌ Failed to start run: {resp.status_code}")
                logger.error(resp.text)
                return None
                
        except Exception as e:
            logger.error(f"❌ Error starting run: {e}")
            return None


async def monitor_run(run_id: str, timeout_seconds: int = 600):
    """Monitor run progress"""
    start_time = time.time()
    last_attempts = 0
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Check status
                resp = await client.get(f"{TEST_CONFIG['api_base_url']}/status")
                if resp.status_code == 200:
                    status = resp.json()
                    
                    # Extract key metrics
                    completed = status.get("completed_attempts", 0)
                    total_cost = status.get("total_cost_usd", 0)
                    findings = status.get("novel_findings", 0)
                    cache_hits = status.get("cache_hits", 0)
                    state = status.get("state", "unknown")
                    
                    # Calculate rate
                    elapsed = time.time() - start_time
                    rate = (completed - last_attempts) / 10 if elapsed > 0 else 0
                    last_attempts = completed
                    
                    # Log progress
                    logger.info(
                        f"Progress: {completed}/{TEST_CONFIG['num_attempts']} attempts | "
                        f"Cost: ${total_cost:.4f} | "
                        f"Findings: {findings} | "
                        f"Cache: {cache_hits} | "
                        f"Rate: {rate:.1f}/s | "
                        f"State: {state}"
                    )
                    
                    # Check completion
                    if state == "COMPLETED" or completed >= TEST_CONFIG["num_attempts"]:
                        logger.info("✅ Run completed!")
                        return status
                    
                    # Check timeout
                    if elapsed > timeout_seconds:
                        logger.warning(f"⏱️ Run timed out after {timeout_seconds} seconds")
                        
                        # Try to pause the run
                        await client.post(f"{TEST_CONFIG['api_base_url']}/control/pause")
                        return status
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)


async def get_findings(run_id: str):
    """Get findings from the run"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{TEST_CONFIG['api_base_url']}/findings/{run_id}")
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Failed to get findings: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting findings: {e}")
            return []


async def run_e2e_test():
    """Run the full E2E test"""
    print("\n" + "="*60)
    print("🚀 E2E TEST: 100+ Attempts against Llama-8B")
    print("="*60)
    
    # Check server
    if not await check_server_health():
        return False
    
    # Start run
    run_id = await start_run()
    if not run_id:
        return False
    
    # Monitor progress
    logger.info("Monitoring run progress...")
    final_status = await monitor_run(run_id, timeout_seconds=600)
    
    # Get findings
    findings = await get_findings(run_id)
    
    # Print results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Completed: {final_status.get('completed_attempts', 0)}/{TEST_CONFIG['num_attempts']} attempts")
    print(f"Total Cost: ${final_status.get('total_cost_usd', 0):.4f}")
    print(f"Novel Findings: {len(findings)}")
    print(f"Cache Hits: {final_status.get('cache_hits', 0)}")
    
    if final_status.get('evaluator'):
        eval_stats = final_status['evaluator']
        print(f"\nEvaluation Performance:")
        print(f"  Cache Hit Rate: {eval_stats.get('cache_hit_rate_percent', 0):.1f}%")
        print(f"  Judge Escalation: {eval_stats.get('escalation_rate_percent', 0):.1f}%")
        print(f"  Avg Time: {eval_stats.get('avg_evaluation_time_ms', 0):.0f}ms")
        print(f"  Cost Savings: {eval_stats.get('estimated_cost_savings_percent', 0):.1f}%")
    
    if findings:
        print(f"\nTop Findings:")
        for i, finding in enumerate(findings[:5], 1):
            print(f"  {i}. {finding.get('category', 'unknown')} - "
                  f"Severity: {finding.get('severity', 'unknown')} - "
                  f"Confidence: {finding.get('confidence', 0):.2f}")
    
    # Save detailed report
    report_path = Path(f"data/e2e_api_report_{run_id}.json")
    report = {
        "run_id": run_id,
        "config": TEST_CONFIG,
        "final_status": final_status,
        "findings": findings
    }
    
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved to: {report_path}")
    
    return True


async def quick_test():
    """Run a quick test with just 10 attempts"""
    print("\n🧪 Running quick test (10 attempts)...")
    
    # Temporarily modify config
    original_attempts = TEST_CONFIG["num_attempts"]
    TEST_CONFIG["num_attempts"] = 10
    TEST_CONFIG["cost_cap_usd"] = 0.10
    
    success = await run_e2e_test()
    
    # Restore config
    TEST_CONFIG["num_attempts"] = original_attempts
    TEST_CONFIG["cost_cap_usd"] = 1.0
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E2E Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick test (10 attempts)")
    parser.add_argument("--full", action="store_true", help="Run full test (100+ attempts)")
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(quick_test())
    elif args.full:
        asyncio.run(run_e2e_test())
    else:
        print("Please specify --quick for 10 attempts or --full for 100+ attempts")
        print("\nExamples:")
        print("  uv run scripts/test_e2e_api.py --quick   # Quick test")
        print("  uv run scripts/test_e2e_api.py --full    # Full 100+ attempt test")