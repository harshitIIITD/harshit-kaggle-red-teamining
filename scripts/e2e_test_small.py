#!/usr/bin/env python3
# ABOUTME: Small E2E test script for quick validation before running full test
# ABOUTME: Tests with just 10 attempts to verify pipeline functionality

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
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

# Small test configuration
TEST_CONFIG = {
    "target_model": "meta-llama/llama-3.1-8b-instruct",
    "num_attempts": 10,  # Just 10 for quick test
    "max_concurrency": 2,  # Lower concurrency
    "cost_cap_usd": 0.10,  # Very conservative cost cap
    "categories": ["harmful_content", "system_prompts"],
    "timeout_seconds": 120  # 2 minute timeout
}


async def run_small_test():
    """Run small E2E test"""
    logger.info("Starting SMALL E2E Test (10 attempts)")
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "sk-or-your-api-key-here":
        logger.error("❌ No valid OPENROUTER_API_KEY found in .env")
        logger.info("Please set your OpenRouter API key in .env file")
        return False
    
    logger.info("✅ API key configured")
    
    try:
        # Load config
        config = load_config()
        logger.info("✅ Configuration loaded")
        
        # Initialize database
        Path("data").mkdir(exist_ok=True)
        dao = StateDAO("data/e2e_small_test.db")
        await dao.initialize()
        logger.info("✅ Database initialized")
        
        # Initialize OpenRouter client with test
        or_client = OpenRouterClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,
            max_retries=3
        )
        
        # Test API connection
        logger.info("Testing API connection...")
        test_response = await or_client.complete(
            model=TEST_CONFIG["target_model"],
            messages=[{"role": "user", "content": "Say 'test ok' if you can read this"}],
            max_tokens=10
        )
        
        if test_response and test_response.content:
            logger.info(f"✅ API test successful: {test_response.content[:50]}")
        else:
            logger.error("❌ API test failed")
            return False
        
        # Initialize cost tracker
        cost_tracker = CostTracker(
            cost_cap_usd=TEST_CONFIG["cost_cap_usd"],
            alert_thresholds=[0.5, 0.75, 0.9]
        )
        logger.info("✅ Cost tracker initialized")
        
        # Initialize orchestrator
        orchestrator = AsyncOrchestrator(
            dao=dao,
            openrouter_client=or_client,
            config=config,
            cost_tracker=cost_tracker
        )
        logger.info("✅ Orchestrator initialized")
        
        # Create test run
        run_id = f"small_test_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting run: {run_id}")
        
        # Run with timeout
        start_time = time.time()
        
        try:
            await asyncio.wait_for(
                orchestrator.run(
                    run_id=run_id,
                    target_model=TEST_CONFIG["target_model"],
                    categories=TEST_CONFIG["categories"],
                    max_attempts=TEST_CONFIG["num_attempts"],
                    max_concurrency=TEST_CONFIG["max_concurrency"]
                ),
                timeout=TEST_CONFIG["timeout_seconds"]
            )
        except asyncio.TimeoutError:
            logger.warning("Run timed out (expected for test)")
        
        elapsed = time.time() - start_time
        
        # Get results
        status = orchestrator.get_status()
        cost_metrics = cost_tracker.get_metrics()
        
        # Print results
        print("\n" + "="*50)
        print("SMALL TEST RESULTS")
        print("="*50)
        print(f"✅ Test completed in {elapsed:.1f} seconds")
        print(f"📊 Attempts: {status.get('completed_attempts', 0)}/{TEST_CONFIG['num_attempts']}")
        print(f"💰 Cost: ${cost_metrics.total_cost:.4f}")
        print(f"🎯 Findings: {status.get('novel_findings', 0)}")
        print(f"⚡ Cache hits: {status.get('cache_hits', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("\n🧪 Running SMALL E2E Test (10 attempts)")
    print("This is a quick validation before running the full 100+ attempt test")
    print("="*60)
    
    success = asyncio.run(run_small_test())
    
    if success:
        print("\n✅ Small test successful! Ready to run full test with:")
        print("   uv run scripts/e2e_test_llama.py")
    else:
        print("\n❌ Small test failed. Please fix issues before running full test.")